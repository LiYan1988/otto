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
        
    # n_estimators=600, max_depth=21, max_features=0.9
    # max_depth=33, max_features=0.85, n_estimators=600: 0.49146
    # max_depth=39, max_features=0.85, n_estimators=700: 0.48785
    # max_depth=42, max_features=0.7, n_estimators=700: 0.48868
    n_cv = 3
    ext = ensemble.ExtraTreesClassifier(class_weight='balanced', n_jobs=7)
    params = {}
    params['n_estimators'] = [500, 600, 650, 700, 800] 
#    params['criterion'] = ['gini', 'entropy']
    params['max_depth'] = [30, 33, 36, 39, 42, 45, 48] # this is better larger
#    params['min_samples_split'] = [1, 2, 16] # better smaller
#    params['min_samples_leaf'] = [1, 5, 10] # this is better 1
    params['max_features'] = [0.7, 0.75, 0.8, 0.85, 0.9, 1.0] # better smaller
    kf = cross_validation.StratifiedKFold(y_train, n_folds=n_cv, shuffle=True, 
        random_state=0)
    
    rndcv = model_selection.RandomizedSearchCV(ext, params, n_iter=45,
        scoring='neg_log_loss', cv=kf, verbose=10, random_state=0)
    rndcv.fit(x_train_tfidf, y_train)
    search_results = pd.DataFrame(rndcv.cv_results_)
    search_results.to_csv('ext_randomSearchCV4.csv')
