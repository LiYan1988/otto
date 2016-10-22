# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 12:58:56 2016

@author: lyaa

Transform data and save results
"""

from ottoStart import *

if __name__=='__main__':
    x_train, y_train, x_test = load_data()
    x_train_tfidf, x_test_tfidf, x_train_log, x_test_log = \
        add_features(x_train, x_test)
        
    n_cv = 3
    xgbclf = xgb.XGBClassifier(objective='multi:softprob', silent=False, 
        seed=0, nthread=-1, gamma=1, subsample=0.8, learning_rate=0.3)
    params = {}
#    params['learning_rate'] = [0.01, 0.02, 0.05, 0.1]
    params['n_estimators'] = [60, 100, 140, 180, 200]
    params['max_depth'] = [9, 12, 15, 18, 21]
    params['colsample_bytree'] = [0.3, 0.6, 0.75, 0.9, 1]
    params['min_child_weight'] = [1, 5, 10, 12]
    params['gamma'] = [1, 2, 3, 4]
    kf = cross_validation.StratifiedKFold(y_train, n_folds=n_cv, shuffle=True, 
        random_state=0)
    
    rndcv = model_selection.RandomizedSearchCV(xgbclf, params, n_iter=30,
        scoring='neg_log_loss', cv=kf, verbose=10, random_state=0)
    rndcv.fit(x_train_tfidf, y_train)
    search_results = pd.DataFrame(rndcv.cv_results_)
    search_results.to_csv('xgboost_randomSearchCV.csv')