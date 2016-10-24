# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 22:15:09 2016

@author: lyaa

LogisticRegressionCV gives wrong scores...
"""

from ottoStart import *

if __name__=='__main__':
    x_train, y_train, x_test = load_data()
    x_train_tfidf, x_test_tfidf, x_train_log, x_test_log = \
        add_features(x_train, x_test)
        
#    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True, 
#        random_state=0)
#    lr = linear_model.LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1, 10], 
#        class_weight='balanced', cv=kf, scoring='neg_log_loss', max_iter=1000, 
#        n_jobs=7, multi_class='multinomial', random_state=0, verbose=10)
#    
#    lr.fit(sparse.csr_matrix(x_train), y_train)
#    results = dict(zip(lr.Cs_, np.mean(lr.scores_[0], 0)))
# best: C=0.06
# {0.0001: -1.6611624031762466,
# 0.00077426368268112698: -1.5615927840014849,
# 0.0059948425031894088: -1.5133593103097329,
# 0.046415888336127774: -1.521600763854843,
# 0.35938136638046259: -1.5451347914114772,
# 2.7825594022071258: -1.5546280954065406,
# 21.544346900318821: -1.5550737233783609,
# 166.81005372000558: -1.5553063082048229,
# 1291.5496650148827: -1.5554801711732502,
# 10000.0: -1.5554839198806198}

#    lr = linear_model.LogisticRegression(C=0.06, class_weight='balanced', 
#        max_iter=1000, n_jobs=7, multi_class= 'multinomial', random_state=0, 
#        verbose=10, solver='sag')
#    lr.fit(x_train_log, y_train)
#    y_pred_lr_log = lr.predict_proba(x_test_log)
#    save_data(y_pred_lr_log, 'layer1_lr_log.pkl')


lr = linear_model.LogisticRegression(C=0.1)
lr.fit(x_train_log, y_train)
y = lr.predict_proba(x_test_log)
save_submission(y, 'lr_submission.csv')