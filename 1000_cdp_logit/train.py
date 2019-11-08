import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
'''for implementing simple logisticregression'''
import sklearn
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve,roc_auc_score,auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
'''for chi'''
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from joblib import dump
import random
import gc

class OmopParser(object):
    '''this structures the omop dataset'''
    def __init__(self):
        self.name = 'omop_parser'

    def add_prediction_date(self):
        '''given a patient's visit records, this function returns the prediction_date '''
        '''and whether this patient has a death record (1) or not(0)'''
        '''output is a reduced visit file'''
        visit = pd.read_csv('/train/visit_occurrence.csv')
        cols = ['person_id','visit_start_date']
        visit = visit[cols]
        death = pd.read_csv('/train/death.csv')
        cols = ['person_id','death_date']
        death = death[cols]
        visit_death = pd.merge(death,visit,on = ['person_id'],how = 'inner')
        visit_death['death_date'] = pd.to_datetime(visit_death['death_date'], format = '%Y-%m-%d')
        visit_death['visit_start_date'] = pd.to_datetime(visit_death['visit_start_date'], format = '%Y-%m-%d')
        visit_death['last_visit_death'] = visit_death['death_date'] - visit_death['visit_start_date']
        visit_death['last_visit_death'] = visit_death['last_visit_death'].apply(lambda x: x.days)
        visit_death = visit_death.loc[visit_death['last_visit_death'] <= 180]
        visit_death = visit_death.drop_duplicates(subset = ['person_id'], keep = 'first')
        visit_death = visit_death[['person_id','visit_start_date']]
        visit_death.columns = ['person_id','prediction_date']
        visit_death['death'] = np.ones(visit_death.shape[0])
        visit_live = visit[~visit.person_id.isin(visit_death.person_id)]
        visit_live = visit_live[['person_id','visit_start_date']]
        live_id = visit_live[['person_id']].drop_duplicates(keep = 'first')
        '''
        for patients in the negative case, select patients' latest visit record
        '''
        visit_live = visit_live.sort_values(['person_id','visit_start_date'],ascending = False).groupby('person_id').head(1)
        visit_live = visit_live[['person_id','visit_start_date']]
        visit_live.columns = ["person_id", "prediction_date"]
        visit_live['death'] = np.zeros(visit_live.shape[0])
        person = pd.concat([visit_death,visit_live],axis = 0)
        ''' "person_id","prediction_date","death"'''
        #print("prediction_date, person, death table added")
        person.to_csv('/scratch/train_cleaned_prediction_date.csv', index = False)

    def add_cdp(self,person,k):
        fields = ['person_id','condition_concept_id']
        condition = pd.read_csv('/train/condition_occurrence.csv',usecols = fields)
        condition.drop_duplicates(subset = ['person_id','condition_concept_id'],inplace = True)
        condition.rename(columns = {"condition_concept_id" : "concept_id"}, inplace = True)
        #print("condition data loaded.")
        fields = ['person_id','procedure_concept_id']
        procedure = pd.read_csv('/train/procedure_occurrence.csv',usecols = fields)
        procedure.drop_duplicates(subset = ['person_id','procedure_concept_id'],inplace = True)
        procedure.rename(columns = {"procedure_concept_id" : "concept_id"}, inplace = True)
        #print("procedure data loaded.")
        fields = ['person_id','drug_concept_id']
        drug = pd.read_csv('/train/drug_exposure.csv',usecols = fields)
        drug.drop_duplicates(subset = ['person_id','drug_concept_id'],inplace = True)
        drug.rename(columns = {"drug_concept_id" : "concept_id"}, inplace = True)
        #print("drug data loaded.")
        person = pd.read_csv('/scratch/train_cleaned_prediction_date.csv')
        cdp = pd.concat([condition,drug], axis = 0)
        cdp = pd.concat([cdp,procedure], axis = 0)
        concept = cdp[['concept_id']].groupby(['concept_id']).size().nlargest(k).reset_index()
        concept_list = concept['concept_id'].to_list()
        print("selected feature number",flush = True)
        print(len(concept_list),flush = True)
        with open("/scratch/concept_list.txt", "wb") as fp:   #Pickling
            pickle.dump(concept_list, fp)
        cdp_person_id = cdp['person_id'].drop_duplicates(keep = "first")
        cdp = cdp.merge(cdp_person_id, on = ['person_id'], how = "right")
        cdp['record'] = np.ones(cdp.shape[0])
        cdp.to_csv('/scratch/person_cdp.csv', index = False)
        reader_cdp = pd.read_csv('/scratch/person_cdp.csv',chunksize = 100000)
        pivot_cdp = pd.DataFrame()
        count = 0
        for x in reader_cdp:
            x = x[x['concept_id'].isin(concept_list)]
            pivot_cdp_buffer = x.pivot_table(index = 'person_id',columns = 'concept_id',values = 'record')
            pivot_cdp_buffer.fillna(0,inplace = True)
            count = count + 1
            print("still pivoting" + str(count),flush = True)
            pivot_cdp = pd.concat([pivot_cdp, pivot_cdp_buffer], sort = False)
            x = ''
            pivot_cdp_buffer = ''
        pivot_cdp.fillna(0, inplace = True)
        print("pivoting is done", flush = True)
        person_id =  pd.DataFrame(pivot_cdp.index, columns = ['person_id'])
        death = pd.merge(person_id,person, on = ['person_id'], how = 'left')
        return pivot_cdp, death, concept_list

    def training(self,pivot_cdp,death,concept_list):
        X = pivot_cdp.loc[:,concept_list]
        Y = death[['death']]
        X.fillna(0, inplace = True)
        Y.fillna(0, inplace = True)
        n_cols = X.shape[1]
        X = np.array(X)
        Y = np.array(Y)
        model = LogisticRegression(penalty = 'l1', tol = 0.0001,random_state = None, max_iter = 10000).fit(X,Y)
        dump(model,'/model/baseline.joblib')
        print("training models trained")



if __name__ == '__main__':
    print("run process_omop.py",flush = True)
    op = OmopParser()
    person = op.add_prediction_date()
    pivot_cdp,death,concept_list = op.add_cdp(person,1000)
    op.training(pivot_cdp,death,concept_list)
    print("training finished")
