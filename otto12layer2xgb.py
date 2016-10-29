# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 20:23:20 2016

@author: lyaa

Train xgboost on KNN features from layer 1. This is the second layer
"""

from ottoStart import *


x_train, y_train, x_test = load_data()
#x_train_tfidf, x_test_tfidf, x_train_log, x_test_log = \
#    add_features(x_train, x_test)

# load data 
x_train_knn = []
x_test_knn = []
for i in range(10):
    x_train_knn.append(read_data(
        'layer1/megaknn_braycurtis_train_pred_model_{}.pkl'.format(i)))
    x_test_knn.append(read_data(
        'layer1/megaknn_braycurtis_test_pred_model_{}.pkl'.format(i)))
    
x_train_knn = np.hstack(tuple(x_train_knn))
x_test_knn = np.hstack(tuple(x_test_knn))

xgbclf = xgb.XGBClassifier(objective='multi:softprob', silent=False, 
    seed=0, nthread=-1, gamma=10, subsample=0.8, learning_rate=0.1, 
    n_estimators=10, max_depth=3, colsample_bytree=1, min_child_weight=1)
cv_scores = model_selection.cross_val_score(xgbclf, x_train_knn, y_train, 
    scoring='neg_log_loss', cv=4, verbose=10)
#xgbclf.fit(x_train_knn, y_train)
#y_pred_knn = xgbclf.predict_proba(x_test_knn)