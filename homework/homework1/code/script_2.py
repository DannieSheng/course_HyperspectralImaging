#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 16:32:02 2018

@author: hudanyun.sheng
"""

import numpy as np
import PCT
 
# create a covariance matrix
B = 3
d = 2
R = np.random.randn(B,B)
#R = np.random.uniform(low=0.0, high=1, size=(B, B))
C = np.dot(R, np.transpose(R))


# sample data from the zero-mean Gaussian with the covariance matrix defined above
mu = np.zeros(B)
x = np.random.multivariate_normal(mu, C, 1000)


# perform principal component transform on dataset x
w, y, cov_mat, eigVals, eigVecs = PCT.PCT(x, d)

# calculate the true variance of the projected data (i.e. y)
cov_maty = np.cov(np.transpose(y))

# calculate the difference of the true variance of the projected data and the eigenvalues
diff = np.diag(cov_maty)-eigVals[0:d]
print(diff)