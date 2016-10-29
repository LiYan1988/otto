# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 19:18:31 2016

@author: lyaa
"""

from ottoStart import *

x_train, y_train, x_test = load_data()
x1, x2, y1, y2 = model_selection.train_test_split(x_train, y_train, 
    test_size=0.8, random_state=0)

# braycurtis
#n_neighbors = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
#estimators = []
#for nn in n_neighbors:
#        estimators.append(neighbors.KNeighborsClassifier(n_neighbors=nn,
#            n_jobs=-1, weights='distance', metric='braycurtis'))
#megaknn = MegaClassifier(estimators, cv=5)
#megaknn.fit_predict_proba(x_train.values, y_train.values, x_test.values, 
#    random_state=0)
#megaknn.save_results('megaknn_braycurtis')

# euclidean
n_neighbors = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
estimators = []
for nn in n_neighbors:
    estimators.append(neighbors.KNeighborsClassifier(n_neighbors=nn,
        n_jobs=-1, weights='distance', metric='euclidean'))
megaknn = MegaClassifier(estimators, cv=5)
megaknn.fit_predict_proba(x_train.values, y_train.values, x_test.values, 
    random_state=0)
megaknn.save_results('layer1/megaknn_euclidean')

# manhattan
#n_neighbors = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
#estimators = []
#for nn in n_neighbors:
#    estimators.append(neighbors.KNeighborsClassifier(n_neighbors=nn,
#        n_jobs=-1, weights='distance', metric='manhattan'))
#megaknn = MegaClassifier(estimators, cv=5)
#megaknn.fit_predict_proba(x_train.values, y_train.values, x_test.values, 
#    random_state=0)
#megaknn.save_results('layer1/megaknn_manhattan')