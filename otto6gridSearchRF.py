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
        
    # max_depth=21, max_features=0.6, n_estimators=600
    # max_depth=33, max_features=0.5, n_estimators=800: 0.50777
    # max_depth=39, max_features=0.5, n_estimators=950: 0.50339 ## the best ##
    # max_depth=42, max_features=0.4, n_estimators=900: 0.50364
    # max_depth=33, max_features=0.4, n_estimators=900: 0.50375
    # max_depth=39, max_features=0.3, n_estimators=800: 0.50437
    n_cv = 3
    rf = ensemble.RandomForestClassifier(class_weight='balanced', n_jobs=7)
    params = {}
    params['n_estimators'] = [800, 850, 900, 950, 1000]
#    params['criterion'] = ['gini', 'entropy']
    params['max_depth'] = [33, 36, 39, 42, 45] # larger better
#    params['min_samples_split'] = [2, 8, 16, 32] # smaller better
#    params['min_samples_leaf'] = [1, 5, 10] # smaller better
    params['max_features'] = [0.3, 0.4, 0.5, 0.6] # smaller better
    kf = cross_validation.StratifiedKFold(y_train, n_folds=n_cv, shuffle=True, 
        random_state=0)
    
    rndcv = model_selection.RandomizedSearchCV(rf, params, n_iter=30,
        scoring='neg_log_loss', cv=kf, verbose=10, random_state=0)
    rndcv.fit(x_train_tfidf, y_train)
    search_results = pd.DataFrame(rndcv.cv_results_)
    search_results.to_csv('rf_randomSearchCV4.csv')
