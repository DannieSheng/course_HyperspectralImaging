#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 21:37:20 2018

@author: hudanyun.sheng
"""

import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import PCT
plt.close("all")
dataFile = '/Users/hudanyun.sheng/Google Drive/Me/201808Fall/ENV6932/homework/homework1/imagingspectroscopy/AsterThings.mat'
data     = scio.loadmat(dataFile)
Aster5 = data['Aster5']
Aster5Names = data['Aster5Names']
Aster5Wvs   = data['Aster5Wvs']

for i in range(0,len(Aster5Names)):
    plt.plot(np.transpose(Aster5Wvs), Aster5[:,i], label = Aster5Names[i][0][0])
    plt.title('Spectra')
    plt.xlabel('Wavelength')
    plt.ylabel('Spectra')
plt.legend()
plt.savefig('/Users/hudanyun.sheng/Google Drive/Me/201808Fall/ENV6932/homework/homework1/code/1b/Spectra.png')

numE = 3
numP = 5000
#alpha = np.array([1, 2, 3])
alpha = np.linspace(1,numE,num=numE)
P = np.random.dirichlet(alpha, size=numP)
E = Aster5[:, 0:numE]
X = np.empty([numP, np.shape(Aster5)[0]])
for i in range(0, numP):
    X[i] = np.sum(E*P[i], axis = 1)

mu = np.zeros(np.shape(Aster5)[0])
temp = np.random.uniform(low=0.0, high=0.1, size=(np.shape(Aster5)[0], np.shape(Aster5)[0]))
sigma = np.dot(temp, np.transpose(temp))
N = np.random.multivariate_normal(mu, sigma, 5000)
S = X+N

d = 3
w, projectedData, cov_mat, eigVals, eigVecs = PCT.PCT(S, d)
MAXpd = math.ceil(np.max(abs(projectedData)))
fig1 = plt.figure()
fig1 = Axes3D(fig1)
fig1.scatter(projectedData[:,0], projectedData[:,1], projectedData[:,2], marker = '.', linewidth = 0.01)
fig1.set_xlim([-MAXpd, MAXpd])
fig1.set_ylim([-MAXpd, MAXpd])
fig1.set_zlim([-MAXpd, MAXpd])
plt.title('Scatter Plot of the Projected Data when m=' + str(numE))
plt.show
plt.savefig('/Users/hudanyun.sheng/Google Drive/Me/201808Fall/ENV6932/homework/homework1/code/1b/Scatter' + str(numE) + '.png')

fig2 = plt.figure()
fig2 = plt.plot(np.arange(1, len(eigVals)+1, 1), eigVals, 'bv', markersize=1)
plt.title('Eigenvalues of the covariance matrix when m=' + str(numE))
plt.savefig('/Users/hudanyun.sheng/Google Drive/Me/201808Fall/ENV6932/homework/homework1/code/1b/Eigenvalue' + str(numE) +'.png')
