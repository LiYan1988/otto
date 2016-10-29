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
from scipy import sparse, io, optimize
import matplotlib.pyplot as plt
import cPickle as pickle

import xgboost as xgb
from sklearn import (preprocessing, manifold, decomposition, ensemble,
                     feature_extraction, model_selection, cross_validation,
                     calibration, linear_model, metrics, neighbors)
from sklearn.base import BaseEstimator, ClassifierMixin

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
    
class MegaClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators, cv=4):
        self.estimators = estimators
        self.cv = cv
        
    def split(self, X, y, random_state=0):
        self.train_indexes = []
        self.valid_indexes = []
        kf = model_selection.StratifiedKFold(n_splits=self.cv, shuffle=True, 
            random_state=random_state)
        for train_index, valid_index in kf.split(X, y):
            self.train_indexes.append(train_index)
            self.valid_indexes.append(valid_index)
            
    def calc_blending_logloss(self, weights):
        log_losses = []
        for i, valid_pred in enumerate(self.y_train_pred_cv):
            y_final_pred_prob = None
            for j, est_pred in enumerate(valid_pred):
                if y_final_pred_prob is None:
                    y_final_pred_prob = weights[j]*est_pred
                else:
                    y_final_pred_prob = np.add(y_final_pred_prob, 
                        (weights[j]*est_pred))
            log_losses.append(metrics.log_loss(self.y_train_true_cv[i], 
                y_final_pred_prob))
        log_losses = np.array(log_losses)
        
        return log_losses.mean()
        
    def fit_predict_proba(self, X, y, X_test, random_state=0):
        self.y_train_pred_cv = [] # prediction on split train set
        self.y_train_true_cv = [] # ground truth labels on split train set
        self.y_test_pred_cv = [] # prediction on test set by cv trained models
        self.split(X, y, random_state=random_state)
        
        for i in range(self.cv):
            train_index = self.train_indexes[i]
            valid_index = self.valid_indexes[i]
            X_train = X[train_index]
            y_train = y[train_index]
            X_valid = X[valid_index]
            y_valid = y[valid_index]
            self.y_train_true_cv.append(y_valid)
            
            valid_pred = []
            test_pred = []
            for j, estimator in enumerate(self.estimators):
#                print 'Fitting fold {} of {}'.format(i, 
#                    estimator.__str__().split('(')[0])
                print 'Fitting model {} on fold {}...'.format(j, i)
                estimator.fit(X_train, y_train)
                print 'Predict model {} on fold {}...'.format(j, i)
                valid_pred.append(estimator.predict_proba(X_valid))
                print 'Predict model {} on test data'.format(j)
                test_pred.append(estimator.predict_proba(X_test))
            self.y_train_pred_cv.append(valid_pred)
            self.y_test_pred_cv.append(test_pred)
            
        # Optimize weights
        initial_weights = [1.0/float(len(self.estimators)) for i in 
            range(len(self.estimators))]
        bounds = [(0, 1) for i in range(len(self.estimators))]
        constraints = {'type': 'eq', 'fun': lambda w: 1-sum(w)}
        res = optimize.minimize(self.calc_blending_logloss, initial_weights,
            bounds=bounds, constraints=constraints)
        self.final_weights = res.x
        self.weight_optimize_res = res
        
        # output results of different estimators
        self.y_test_pred_all = []
        for i in range(len(self.estimators)):
            tmp = self.y_test_pred_cv[0][i]
            for j in range(1, self.cv):
                tmp = np.add(tmp, self.y_test_pred_cv[j][i])
            tmp = tmp/float(self.cv)
            self.y_test_pred_all.append(tmp)
            
        # weighted output
        self.y_test_pred_weighted = None
        for i, test_pred in enumerate(self.y_test_pred_all):
            if self.y_test_pred_weighted is None:
                self.y_test_pred_weighted = self.final_weights[i]*test_pred
            else:
                self.y_test_pred_weighted = np.add(self.y_test_pred_weighted, 
                    (self.final_weights[i]*test_pred))
                
        # reshape y_train_pred_cv
        self.y_train_pred_all = []
        for i in range(len(self.estimators)):
            tmp = (self.y_train_pred_cv[j][i] for j in range(self.cv))
            self.y_train_pred_all.append(np.vstack(tmp))
                
        return self.y_test_pred_weighted
        
    def save_results(self, head):
        # save these data for the next layer
        for i in range(len(self.estimators)):
            save_data(self.y_train_pred_all[i], 
                 head+'_train_pred_model_'+str(i)+'.pkl')
            save_data(self.y_test_pred_all[i],
                 head+'_test_pred_model_'+str(i)+'.pkl')
        save_submission(self.y_test_pred_weighted, head+'_test_weighted.csv')
    
if __name__=='__main__':
    x_train, y_train, x_test = load_data()
    x1, x2, y1, y2 = model_selection.train_test_split(x_train, y_train, 
        test_size=0.8, random_state=0)
    
    n_neighbors = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
#    n_neighbors = [2, 4]
    estimators = []
    for nn in n_neighbors:
        estimators.append(neighbors.KNeighborsClassifier(n_neighbors=nn,
            n_jobs=-1))
    megaknn = MegaClassifier(estimators, cv=5)
#    megaknn.fit_predict_proba(x1.values, y1.values, x2.values, 
#        random_state=0)
    megaknn.fit_predict_proba(x_train.values, y_train.values, x_test.values, 
        random_state=0)
    megaknn.save_results('megaknn')