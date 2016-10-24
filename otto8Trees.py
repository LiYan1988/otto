# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 21:19:49 2016

@author: lyaa
"""

from ottoStart import *

if __name__=='__main__':
    x_train, y_train, x_test = load_data()
    x_train_tfidf, x_test_tfidf, x_train_log, x_test_log = \
        add_features(x_train, x_test)
    
    # EXT
    ext = ensemble.ExtraTreesClassifier(class_weight='balanced', n_jobs=7, 
        n_estimators=200, max_depth=21, max_features=0.8, )
    calibrated_ext = calibration.CalibratedClassifierCV(ext, method='isotonic',
        cv=7)
    calibrated_ext.fit(x_train_tfidf, y_train)
    y_pred_ext_tfidf = calibrated_ext.predict_proba(x_test_tfidf)
    save_submission(y_pred_ext_tfidf, 'layer1_ext_tfidf.csv')
    
    # RF
    rf = ensemble.RandomForestClassifier(class_weight='balanced', n_jobs=7, 
        n_estimators=400, max_depth=21, max_features=0.9, )
    calibrated_rf = calibration.CalibratedClassifierCV(rf, method='isotonic',
        cv=7)
    calibrated_rf.fit(x_train_tfidf, y_train)
    y_pred_rf_tfidf = calibrated_rf.predict_proba(x_test_tfidf)
    save_submission(y_pred_rf_tfidf, 'layer1_rf_tfidf.csv')
    
