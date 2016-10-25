# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 11:03:19 2016

@author: lyaa
"""

from ottoStart import *

if __name__=='__main__':
    filename = []
    filename.append('layer1_ext_tfidf.csv')
    filename.append('layer1_rf_tfidf.csv')
    filename.append('layer1_lr_log.csv')
    
    x = []
    for f in filename:
        x.append(pd.read_csv(f))
    