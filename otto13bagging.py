# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 12:20:38 2016

@author: lyaa
"""

from ottoStart import *

x_train, y_train, x_test = load_data()
x_train_tfidf, x_test_tfidf, x_train_log, x_test_log = \
    add_features(x_train, x_test)

# estimators
ext = ensemble.ExtraTreesClassifier(class_weight='balanced', n_jobs=8, 
    n_estimators=500, max_depth=40, max_features=0.7, verbose=0)

rf = ensemble.RandomForestClassifier(class_weight='balanced', n_jobs=8, 
    n_estimators=300, max_depth=40, max_features=0.6, verbose=0)

lr = linear_model.LogisticRegression(C=0.06, class_weight='balanced', 
    max_iter=1000, n_jobs=8, multi_class='multinomial', random_state=0, 
    verbose=100, solver='sag', tol=0.001)

#n_neighbors = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
estimators = [ext, rf, lr]
#for nn in n_neighbors:
#        estimators.append(neighbors.KNeighborsClassifier(n_neighbors=nn,
#            n_jobs=-1, weights='distance', metric='braycurtis'))
#    estimators.append(neighbors.KNeighborsClassifier(n_neighbors=nn,
#        n_jobs=-1, weights='distance', metric='euclidean'))

xgbclf = xgb.XGBClassifier(objective='multi:softprob', silent=False, 
    seed=0, nthread=-1, n_estimators=240, max_depth=15, colsample_bytree=0.8,
    min_child_weight=5, gamma=1, subsample=0.8, learning_rate=0.1)

estimators_layer1 = estimators
estimators_layer2 = [xgbclf]
megabag = MegaBagging(x_train_tfidf.toarray(), y_train.values, 
    x_test_tfidf.toarray(), estimators_layer1, estimators_layer2, cv=5, 
    repetitions_layer2=20)
megabag.cv_repeat(0)

if not os.path.exists('megabag1'):
    os.mkdir('megabag1')
    
save_data(megabag.y_test_pred, 'megabag1/megabag1_final.pkl')
save_data(megabag.train_new_x, 'megabag1/megabag1_train.pkl') 
save_data(megabag.test_new_x, 'megabag1/megabag1_test.pkl')

