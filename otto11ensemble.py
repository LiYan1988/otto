# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 11:03:19 2016

@author: lyaa
"""

from ottoStart import *

if __name__=='__main__':
    _, y_train, _ = load_data()
    filename = []
    filename.append('layer1_ext_tfidf.pkl')
    filename.append('layer1_rf_tfidf.pkl')
    filename.append('layer1_lr_log.pkl')
    
    x = []
    for f in filename:
        x.append(read_data(f))
    x = np.hstack(tuple(x))
    x_train = x[:y_train.shape[0],:]
    x_test = x[y_train.shape[0]:,:]

    n_cv = 3
    xgbclf = xgb.XGBClassifier(objective='multi:softprob', silent=False, 
    seed=0, nthread=-1, gamma=1, subsample=0.8, learning_rate=0.01)
    params = {}
    params['learning_rate'] = [0.01, 0.02, 0.05, 0.1]
    params['n_estimators'] = [160, 200, 300, 400, 500]
    params['max_depth'] = [6, 9, 12, 15, 18, 21]
    params['colsample_bytree'] = [0.8, 0.9, 1]
    params['min_child_weight'] = [1, 2, 5, 7, 10]
    params['gamma'] = [1, 2, 4]
    kf = cross_validation.StratifiedKFold(y_train, n_folds=n_cv, shuffle=True, 
        random_state=0)
    
    rndcv = model_selection.RandomizedSearchCV(xgbclf, params, n_iter=45,
        scoring='neg_log_loss', cv=kf, verbose=10, random_state=0)
    rndcv.fit(x_train, y_train)
    search_results = pd.DataFrame(rndcv.cv_results_)
    search_results.to_csv('xgboost_randomSearchCV_layer2.csv')