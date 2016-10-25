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
    
    x_cali0, x_cali1, y_cali0, y_cali1 = model_selection.train_test_split(
        x_train_tfidf, y_train, test_size=0.2, random_state=0)
    
    # EXT
    print 'Calculating EXT...'
    ext = ensemble.ExtraTreesClassifier(class_weight='balanced', n_jobs=8, 
        n_estimators=700, max_depth=39, max_features=0.85, verbose=10)
    ext.fit(x_cali0, y_cali0)
    calibrated_ext = calibration.CalibratedClassifierCV(ext, method='isotonic',
        cv='prefit')
    calibrated_ext.fit(x_cali1, y_cali1)
    y_hat_ext_tfidf = calibrated_ext.predict_proba(x_train_tfidf)
    y_pred_ext_tfidf = calibrated_ext.predict_proba(x_test_tfidf)
    y_ext_tfidf = np.vstack((y_hat_ext_tfidf, y_pred_ext_tfidf))
    save_data(y_ext_tfidf, 'layer1_ext_tfidf.pkl')
    del ext, calibrated_ext
    
    # RF
    print 'Calculating RF...'
    rf = ensemble.RandomForestClassifier(class_weight='balanced', n_jobs=8, 
        n_estimators=950, max_depth=39, max_features=0.5, verbose=10)
    rf.fit(x_cali0, y_cali0)
    calibrated_rf = calibration.CalibratedClassifierCV(rf, method='isotonic',
        cv='prefit')
    calibrated_rf.fit(x_cali1, y_cali1)
    y_hat_rf_tfidf = calibrated_rf.predict_proba(x_train_tfidf)
    y_pred_rf_tfidf = calibrated_rf.predict_proba(x_test_tfidf)
    y_rf_tfidf = np.vstack((y_hat_rf_tfidf, y_pred_rf_tfidf))
    save_data(y_rf_tfidf, 'layer1_rf_tfidf.pkl')
    del rf, calibrated_rf
    
    #LR
    print 'Calculating LR...'
    lr = linear_model.LogisticRegression(C=0.06, class_weight='balanced', 
        max_iter=100, n_jobs=8, multi_class='multinomial', random_state=0, 
        verbose=10, solver='sag')
    lr.fit(x_train_log, y_train)
    y_hat_lr_log = lr.predict_proba(x_train_log)
    y_pred_lr_log = lr.predict_proba(x_test_log)
    y_lr_log = np.vstack((y_hat_lr_log, y_pred_lr_log))
    save_data(y_lr_log, 'layer1_lr_log.pkl')
