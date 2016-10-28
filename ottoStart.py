# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 16:38:27 2016

@author: lyaa

0.46251, xgboost with max_depth=9, sub_sample=0.9, eta=0.1, n_rounds=200
0.45829, xgboost with max_depth=12, sub_sample=0.9, eta=0.1, n_rounds=200
0.46019, xgboost with max_depth=12, sub_sample=0.8, eta=0.1, n_rounds=250
"""

import pandas as pd
import numpy as np
from scipy import sparse, io
import matplotlib.pyplot as plt
import cPickle as pickle

import xgboost as xgb
from sklearn import (preprocessing, manifold, decomposition, ensemble,
                     feature_extraction, model_selection, cross_validation,
                     calibration, linear_model)


def load_data():
    x_train = pd.read_csv('train.csv')
    x_test = pd.read_csv('test.csv')
    
    y_train = x_train.target
    relabeler = preprocessing.LabelEncoder()
    y_train = pd.Series(data=relabeler.fit_transform(y_train), name='target')
    x_train.drop(['id', 'target'], axis=1, inplace=True)
    x_test.drop(['id'], axis=1, inplace=True)
    
    return x_train, y_train, x_test
    
def save_submission(y_pred_proba, file_name):
    df = pd.DataFrame(data=y_pred_proba, columns=['Class_1', 'Class_2', 
        'Class_3', 'Class_4', 'Class_5', 'Class_6', 'Class_7', 'Class_8', 
        'Class_9'])
    df.index = df.index+1
    df.index.name = 'id'
    df.to_csv(file_name, index=True)

def add_features(x_train, x_test):
    """Add new features 
    """
    x = pd.concat([x_train, x_test])
    tsne = pd.read_csv('tsne3all.csv')
    scaler = preprocessing.StandardScaler(with_mean=False)
    tfidf = feature_extraction.text.TfidfTransformer()
    x_tfidf = tfidf.fit_transform(x)
    x_tfidf = scaler.fit_transform(x_tfidf).toarray()
    x_tfidf = np.hstack((x_tfidf, tsne))
    x_train_tfidf = sparse.csr_matrix(x_tfidf[:x_train.shape[0], :])
    x_test_tfidf = sparse.csr_matrix(x_tfidf[x_train.shape[0]:, :])
    
    scaler = preprocessing.StandardScaler(with_mean=False)
    x_log = np.log10(x+1)
    x_log = scaler.fit_transform(x_log)
    x_log = np.hstack((x_log, tsne))
    x_train_log = sparse.csr_matrix(x_log[:x_train.shape[0], :])
    x_test_log = sparse.csr_matrix(x_log[x_train.shape[0]:, :])

    return x_train_tfidf, x_test_tfidf, x_train_log, x_test_log
    
def save_data(data, file_name):
    """File name must ends with .pkl
    """
    with open(file_name, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        
def read_data(file_name):
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
        
    return data
    
if __name__=='__main__':
    x_train, y_train, x_test = load_data()
    train_mat = xgb.DMatrix(data=x_train, label=y_train)
    test_mat = xgb.DMatrix(data=x_test)
    
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 12
    param['sub_sample'] = 0.9
    param['min_child_weight'] = 10
    param['num_class'] = 9
    param['nthread'] = 7
    param['silent'] = False
    n_rounds = 200
    
    bst = xgb.train(param, train_mat, n_rounds)
    
    y_test = bst.predict(test_mat)
    save_submission(y_test, 'test_submission.csv')