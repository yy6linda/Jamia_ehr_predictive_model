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
'''for implementing neural network'''
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras import regularizers
'''for implementing  RandomForestClassifier'''
from sklearn.ensemble import RandomForestClassifier
'''for implementing XGBOOST'''
import xgboost
'''for chi'''
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
'''for finding feature importance in keras model'''
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import eli5
from eli5.sklearn import PermutationImportance
from joblib import dump


class data_preparation(object):
    def train_preparation(self):
        '''
        This function:
            1. split cleaned data gained from process omop.py to two parts: train(0.8) and test(0.2)
            2. in train set, only clinical information for  patients who have at least 20 concept_id features are drug_selected
            3. in train set, clinical information for patients whose prediction_death date is 3-6 month is not used
        '''

        data = pd.read_csv('/model/train_cleaned_prediction_date_plus_demographic_data_plus_condition.csv',low_memory=False)
        Y = data[['death']]
        X_raw = data.drop(['scaled_age','death','prediction_date','person_id'], axis=1)
        X_raw = X_raw.drop(X_raw.filter(regex='race').columns, axis=1)
        X_raw = X_raw.drop(X_raw.filter(regex='gender').columns, axis=1)
        X_raw.fillna(0,inplace = True)
        #print(list(X_raw.columns.values), flush = True)
        X_demographic= data[['scaled_age']]
        X_demographic = pd.concat([X_demographic,data.filter(regex='race')],axis=1)
        X_demographic = pd.concat([X_demographic,data.filter(regex='gender')],axis=1)
        X_demographic.fillna(0,inplace = True)
        return X_raw, Y, X_demographic

    def feature_selection_chi(self, k,X_raw, Y, X_demographic):

        '''
        chi square scoring function is used to select the top k features.
        '''
        ch2 = SelectKBest(chi2, k)
        X_raw_k = ch2.fit_transform(X_raw, Y)
        sel_features = X_raw.columns[ch2.get_support(indices=True)].tolist()
        print(sel_features)
        X_raw_k= pd.DataFrame(X_raw_k,columns=sel_features)
        X_raw_k = X_raw_k.set_index(X_raw.index)
        X_k_demographic =X_raw_k.join(X_demographic)
        k_features = pd.DataFrame(X_raw_k.columns)
        k_demographic_features = pd.DataFrame(X_k_demographic.columns)
        print("features")
        print(list(X_raw_k.columns.values))
        k_features.to_csv('/model/k_feature.csv',index = False)
        k_demographic_features.to_csv('/model/k_demographic_feature.csv',index = False)
        return X_k_demographic,k_features

    def logit_model(self,X_k_demographic, Y):
        clf = LogisticRegressionCV(cv = 20, penalty='l2', tol=0.0001, fit_intercept=True, intercept_scaling=1,
        class_weight=None, solver ='lbfgs',random_state=None, max_iter=500, verbose=0, n_jobs=None).fit(X_k_demographic, Y)
        dump(clf, '/model/20condition.joblib')


if __name__ == '__main__':
    print("start",flush = True)
    m = data_preparation()
    print("load data", flush = True)
    X_raw, Y, X_demographic = m.train_preparation()
    print("build the model", flush = True)
    X_k_demographic,sel_features = m.feature_selection_chi(20,X_raw, Y, X_demographic)
    m.logit_model(X_k_demographic, Y)
    print("finished training", flush = True)
