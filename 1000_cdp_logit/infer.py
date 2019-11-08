import datetime
import pandas as pd
import numpy as np
from datetime import datetime
'''for implementing simple logisticregression'''
import sklearn
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve,roc_auc_score,auc
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
'''for saving models'''
import pickle
from joblib import load
import random
import gc
ROOT = "/"


class OmopParser(object):
    '''this structures the omop dataset'''
    def __init__(self):
        self.name = 'omop_assembler'

    def add_features(self):
        visit = pd.read_csv('/infer/visit_occurrence.csv')
        visit.drop_duplicates(inplace = True)
        person = pd.read_csv('/infer/person.csv')
        person.drop_duplicates(inplace = True)
        person = person[['person_id']]
        visit = visit[['person_id','visit_start_date']]
        visit.columns = ["person_id", "last_visit_date"]
        person = pd.merge(person, visit, on = ['person_id'], how = 'left')
        fields = ['person_id','condition_concept_id']
        condition = pd.read_csv('/infer/condition_occurrence.csv',usecols = fields)
        condition.drop_duplicates(subset = ['person_id','condition_concept_id'],inplace = True)
        condition.rename(columns = {"condition_concept_id" : "concept_id"}, inplace = True)
        print("condition data loaded.")
        fields = ['person_id','procedure_concept_id']
        procedure = pd.read_csv('/infer/procedure_occurrence.csv',usecols = fields)
        procedure.drop_duplicates(subset = ['person_id','procedure_concept_id'],inplace = True)
        procedure.rename(columns = {"procedure_concept_id" : "concept_id"}, inplace = True)
        print("procedure data loaded.")
        fields = ['person_id','drug_concept_id']
        drug = pd.read_csv('/infer/drug_exposure.csv',usecols = fields)
        drug.drop_duplicates(subset = ['person_id','drug_concept_id'],inplace = True)
        drug.rename(columns = {"drug_concept_id" : "concept_id"}, inplace = True)
        print("drug data loaded.")
        cdp = pd.concat([condition,drug],axis = 0)
        cdp = pd.concat([cdp,procedure],axis = 0)
        cdp_person_id = cdp['person_id'].drop_duplicates(keep = "first")
        cdp = cdp.merge(cdp_person_id, on = ['person_id'], how = "right")
        cdp['record'] = np.ones(cdp.shape[0])
        cdp.to_csv('/scratch/cdp_infer.csv', index = False)
        with open("/scratch/concept_list.txt", "rb") as fp:   # Unpickling
            concept_list = pickle.load(fp)
        reader_cdp = pd.read_csv('/scratch/cdp_infer.csv',chunksize = 100000)
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
        person_id =  pd.DataFrame(pivot_cdp.index, columns = ['person_id']).astype('int')
        person_id_left = visit[~visit['person_id'].isin(list(pivot_cdp.index))]
        person_id_left = person_id_left[['person_id']]
        person_id_left.drop_duplicates(inplace = True)
        return concept_list,pivot_cdp,person_id,person_id_left

    def model(self,concept_list,pivot_cdp,person_id,person_id_left):
        X = pivot_cdp.loc[:,concept_list]
        print('X.shape[0]')
        print(X.shape[0])
        X.fillna(0, inplace = True)
        X = np.array(X)
        clf = load('/model/baseline.joblib')
        ## create a all-zero prediction
        length = len(concept_list)
        predictors_all_zero = pd.DataFrame(np.zeros((1,length)))
        Y_predictors = clf.predict_proba(predictors_all_zero)[:,1]
        Y_pred = clf.predict_proba(X)[:,1]
        output = pd.DataFrame(Y_pred,columns = ['score'])
        output_prob = pd.concat([person_id,output],axis = 1)
        person_id_left['score'] = Y_predictors[0]
        output_prob = pd.concat([output_prob,person_id_left],axis = 0)
        output_prob.drop_duplicates(subset = ['person_id'], keep = 'first',inplace = True)
        output_prob.to_csv('/output/predictions.csv')


if __name__ == '__main__':
    print("start",flush = True)
    op = OmopParser()
    concept_list,pivot_cdp, person_id,person_id_left = op.add_features()
    op.model(concept_list,pivot_cdp, person_id,person_id_left)
