#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 11:15:30 2018

@author: hudanyun.sheng
@email: hdysheng@ufl.edu
"""
import numpy as np

def PCT(X,d):
    mu = np.mean(X, axis = 0)
    X_std = X-mu
    cov_mat = np.cov(np.transpose(X))
    eigVecs, eigVals, vh = np.linalg.svd(cov_mat, full_matrices=True)
#    if ((X==np.transpose(X)).any()):
#    eigVecs, eigVals, _ = np.linalg.svd(cov_mat, full_matrices=True)[0:2]
#    print(np.dot(cov_mat, eigVecs[:,1]))
#    print(eigVals[1]*eigVecs[:,1])
#    else:
#    eigVals, eigVecs = np.linalg.eig(cov_mat)
#    idx = eigVals.argsort()[::-1]   
#    eigVals = eigVals[idx]
#    eigVecs = eigVecs[:,idx]    
    w = eigVecs[:, 0:d]
    projected_data = np.dot(X_std, w);
    return w, projected_data, cov_mat, eigVals, eigVecs