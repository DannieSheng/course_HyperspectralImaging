#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 11:15:30 2018

@author: hudanyun.sheng
"""

#import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import PCT
plt.close("all")

B = 400

if B==2:
    d = 2
else:
    d = 3
R = np.random.randn(B,B)
mu = np.zeros(B)
C = np.dot(np.transpose(R), R)
x = np.random.multivariate_normal(mu, C, 1000)
w, y, cov_mat, eigVals, eigVecs = PCT.PCT(x, d)
MAXy = math.ceil(np.max(abs(y)))
    
if d == 2:
    fig1 = plt.figure()
    fig1 = plt.scatter(y[:,0], y[:,1], s=None, c=None, marker='.', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=0.01, verts=None, edgecolors=None, hold=None, data=None, label = 'data')
#    fig1 = plt.plot(np.linspace(0, eigVals[0], num=10), np.zeros(10), 'r', linewidth = 3, label = 'eigenvalue 1')
#    fig1 = plt.plot(np.zeros(10), np.linspace(0, eigVals[1], num=10), 'y', linewidth = 3, label = 'eigenvalue 2')
    plt.xlim([-MAXy, MAXy])
    plt.ylim([-MAXy, MAXy])
    plt.title('Scatter Plot of the Projected Data when dimension is ' + str(B))
    plt.show()
else:
    fig1 = plt.figure()
    fig1 = Axes3D(fig1)
    fig1.scatter(y[:,0], y[:,1], y[:,2], marker = '.', linewidth = 0.01, label='data')
    fig1.set_xlim([-MAXy, MAXy])
    fig1.set_ylim([-MAXy, MAXy])
    fig1.set_zlim([-MAXy, MAXy])
#    fig1 =plt.plot(np.linspace(0, eigVals[0], num=10), np.zeros(10), np.zeros(10), 'r', linewidth = 3, label = 'eigenvalue 1')
#    fig1 =plt.plot(np.zeros(10), np.linspace(0, eigVals[1], num=10),np.zeros(10), 'g', linewidth = 3, label = 'eigenvalue 2')
#    fig1 =plt.plot(np.zeros(10), np.zeros(10), np.linspace(0, eigVals[2], num=10),'b', linewidth = 3, label = 'eigenvalue 3')
    plt.title('Scatter Plot of the Projected Data')
    plt.show
#plt.legend()
plt.savefig('ScatterPlotProjected' + str(B) + '.png')

fig2 = plt.figure()
fig2 = plt.plot(np.arange(1, len(eigVals)+1, 1), eigVals, 'bv', markersize=5)
plt.title('Eigenvalues of the covariance matrix when dimension is '+ str(B))
plt.savefig('EigenvaluePlotEigenvalue' + str(B) +'.png')
   
# whitening
fudge = 1E-18
D = np.diag(1. / np.sqrt(eigVals[0:d]+fudge))
whitenedData = np.dot(y, D)
MAXwh = math.ceil(np.max(abs(whitenedData)))
if d == 2:
    fig3 = plt.figure() 
    fig3 = plt.scatter(whitenedData[:,0], whitenedData[:,1], s=None, c=None, marker='.', cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=0.01, verts=None, edgecolors=None, hold=None, data=None)
    plt.xlim([-MAXwh, MAXwh])
    plt.ylim([-MAXwh ,MAXwh])
    plt.title('Scattter Plot of Whitened Data')
else:
    fig3 = plt.figure()
    fig3 = Axes3D(fig3)
    fig3.scatter(whitenedData[:,0], whitenedData[:,1], whitenedData[:,2], marker = '.')
    fig3.set_xlim([-MAXwh, MAXwh])
    fig3.set_ylim([-MAXwh, MAXwh])
    fig3.set_zlim([-MAXwh, MAXwh])
    plt.title('Scatter Plot of Whitened Data')
    plt.show
plt.savefig('ScatterPlotWhitened' + str(B) + '.png')