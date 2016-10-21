# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 09:43:02 2016

@author: lyaa

The TSNE in sklearn is buggy, it cannot handle many samples, use R!
"""

from ottoStart import *

if __name__=='__main__':
    x_train, y_train, x_test = load_data()
    
    
    train_mat = xgb.DMatrix(data=x_train, label=y_train)
    test_mat = xgb.DMatrix(data=x_test)
    
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 12
    param['sub_sample'] = 0.9
    param['min_child_weight '] = 10
    param['num_class'] = 9
    param['nthread'] = 7
    param['silent'] = False
    n_rounds = 200
    
    bst = xgb.train(param, train_mat, n_rounds)
    
    y_test = bst.predict(test_mat)
    save_submission(y_test, 'test_submission.csv')