# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 11:03:19 2016

@author: lyaa
"""

from ottoStart import *

if __name__=='__main__':
    x_train, y_train, x_test = load_data()
    x_train_tfidf, x_test_tfidf, x_train_log, x_test_log = \
        add_features(x_train, x_test)
        
    ext = ensemble.ExtraTreesClassifier(class_weight='balanced', n_jobs=8, 
        n_estimators=500, max_depth=40, max_features=0.7, verbose=1)
    
    rf = ensemble.RandomForestClassifier(class_weight='balanced', n_jobs=8, 
        n_estimators=300, max_depth=40, max_features=0.6, verbose=1)
    
    lr = linear_model.LogisticRegression(C=0.06, class_weight='balanced', 
        max_iter=1000, n_jobs=8, multi_class='multinomial', random_state=0, 
        verbose=0, solver='sag')
    
    knn = neighbors.KNeighborsClassifier()
    
    estimators = [ext, rf, lr]
    megaclf = MegaClassifier(estimators, cv=5)
    megaclf.fit_predict_proba(x_train_log, y_train.values, x_test_log, 
        random_state=0)
    megaclf.save_results('megaclf1')