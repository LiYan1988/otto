# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 09:43:02 2016

@author: lyaa
"""

from ottoStart import *

if __name__=='__main__':
    x_train, y_train, x_test = load_data()
    
    n_samples = 15000
    samples = np.random.choice(range(x_train.shape[0]), n_samples)
    x_train_sample = x_train.iloc[:n_samples, :]
    pca = decomposition.PCA(n_components=30)
    x_train_pca = pca.fit_transform(x_train_sample)
    tsne = manifold.TSNE(init='pca', verbose=10)
    train_tsne = tsne.fit_transform(x_train_pca)