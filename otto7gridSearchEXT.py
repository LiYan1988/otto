# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 18:40:52 2016

@author: lyaa
"""

from ottoStart import *

if __name__=='__main__':
    x_train, y_train, x_test = load_data()
    x_train_tfidf, x_test_tfidf, x_train_log, x_test_log = \
        add_features(x_train, x_test)
        
    n_cv = 3
    ext = ensemble.ExtraTreesClassifier(class_weight='balanced', n_jobs=7)
    params = {}
    params['n_estimators'] = [200]
#    params['criterion'] = ['gini', 'entropy']
    params['max_depth'] = [6, 9, 12, 15]
    params['min_samples_split'] = [2, 8, 16, 32]
    params['min_samples_leaf'] = [1, 5, 10]
    params['max_features'] = [0.8, 0.9, 1]
    kf = cross_validation.StratifiedKFold(y_train, n_folds=n_cv, shuffle=True, 
        random_state=0)
    
    rndcv = model_selection.RandomizedSearchCV(ext, params, n_iter=45,
        scoring='neg_log_loss', cv=kf, verbose=10, random_state=0)
    rndcv.fit(x_train_tfidf, y_train)
    search_results = pd.DataFrame(rndcv.cv_results_)
    search_results.to_csv('ext_randomSearchCV.csv')
