# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 12:58:56 2016

@author: lyaa

Transform data and save results
n_estimators=2400, max_depth=18, colsample_bytree=0.8/0.9, min_child_weight=5, 
gamma=2: cv=0.45717
n_estimators=2400, max_depth=15, colsample_bytree=0.8, min_child_weight=5, 
gamma=1: cv=0.45419
n_estimators=3000, max_depth=21, colsample_bytree=0.6, min_child_weight=7, 
gamma=1: cv=0.45477
"""

from ottoStart import *

if __name__=='__main__':
    x_train, y_train, x_test = load_data()
    x_train_tfidf, x_test_tfidf, x_train_log, x_test_log = \
        add_features(x_train, x_test)
        
    n_cv = 3
    xgbclf = xgb.XGBClassifier(objective='multi:softprob', silent=False, 
        seed=0, nthread=-1, gamma=1, subsample=0.8, learning_rate=0.01)
    params = {}
#    params['learning_rate'] = [0.01, 0.02, 0.05, 0.1]
    params['n_estimators'] = [2000, 2400, 3000, 3400]
    params['max_depth'] = [15, 18, 21, 24, 27, 30]
    params['colsample_bytree'] = [0.6, 0.7, 0.8] # smaller the better
    params['min_child_weight'] = [1, 2, 5, 7, 10] # smaller
    params['gamma'] = [0, 1, 2] # smaller
    kf = cross_validation.StratifiedKFold(y_train, n_folds=n_cv, shuffle=True, 
        random_state=0)
    
    rndcv = model_selection.RandomizedSearchCV(xgbclf, params, n_iter=30,
        scoring='neg_log_loss', cv=kf, verbose=10, random_state=0)
    rndcv.fit(x_train_tfidf, y_train)
    search_results = pd.DataFrame(rndcv.cv_results_)
    search_results.to_csv('xgboost_randomSearchCV_eta001_tfidf2.csv')