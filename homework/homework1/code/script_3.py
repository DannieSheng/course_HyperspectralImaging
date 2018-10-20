import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import PCT
plt.close('all')
d = 3

# SanBar Data Set
#dataFile = '/Users/hudanyun.sheng/Google Drive/Me/201808Fall/ENV6932/homework/homework1/imagingspectroscopy/SanBarThings.mat'
#data     = scio.loadmat(dataFile)
#SanBarRGB = data['SanBarRGB']
#SanBarIm88x400 = data['SanBarIm88x400']
#SanBarWvs88x400   = data['SanBarWvs88x400']
#SanBar    = np.reshape(SanBarIm88x400, [np.shape(SanBarIm88x400)[0]*np.shape(SanBarIm88x400)[1], np.shape(SanBarIm88x400)[2]])
#w, SanBarPCT, cov_mat, eigVals, eigVecs  = PCT.PCT(SanBar, d)
#MAXpd = math.ceil(np.max(abs(SanBarPCT)))
#fig1 = plt.figure()
#fig1 = Axes3D(fig1)
#fig1.scatter(SanBarPCT[:,0], SanBarPCT[:,1], SanBarPCT[:,2], marker = '.', linewidth = 0.01)
#fig1.set_xlim([-MAXpd, MAXpd])
#fig1.set_ylim([-MAXpd, MAXpd])
#fig1.set_zlim([-MAXpd, MAXpd])
#plt.title('Scatter Plot of the Projected Data')
#plt.show
#plt.savefig('/Users/hudanyun.sheng/Google Drive/Me/201808Fall/ENV6932/homework/homework1/code/3/ScatterSanBar.png')
#
#fig2 = plt.figure()
#fig2 = plt.plot(np.arange(1, len(eigVals)+1, 1), eigVals, 'bv', markersize=5)
#plt.title('Eigenvalues of the covariance matrix of Sanbarbara')
#plt.savefig('/Users/hudanyun.sheng/Google Drive/Me/201808Fall/ENV6932/homework/homework1/code/3/EigenvaSanbar.png')

# GulfPortCampus Data Set
dataFile = '/Users/hudanyun.sheng/Google Drive/Me/201808Fall/ENV6932/homework/homework1/imagingspectroscopy/GulfPortCampusThings.mat'
data     = scio.loadmat(dataFile)
GulfPortCampusRGB = data['GulfPortCampusRGB']
GulfPortCampusIm  = data['GulfPortCampusIm']
GulfPortCampusWvs = data['GulfPortCampusWvs']
GulfPortOceanMask = data['GulfPortOceanMask']
GP_temp  = np.reshape(GulfPortCampusIm, [np.shape(GulfPortCampusIm)[0]*np.shape(GulfPortCampusIm)[1], np.shape(GulfPortCampusIm)[2]])
GPm      = np.reshape(GulfPortOceanMask, [np.shape(GulfPortOceanMask)[0]*np.shape(GulfPortOceanMask)[1], 1])
GP = np.multiply(GP_temp, np.tile(GPm, np.shape(GP_temp)[1]))
idx = np.where(GP==0)
GP = np.delete(GP, idx[0], axis = 0)
w, GP_PCT, cov_mat, eigVals, eigVecs  = PCT.PCT(GP, d)
MAXpd = math.ceil(np.max(abs(GP_PCT)))
fig1 = plt.figure()
fig1 = Axes3D(fig1)
fig1.scatter(GP_PCT[:,0], GP_PCT[:,1], GP_PCT[:,2], marker = '.', linewidth = 0.01)
fig1.set_xlim([-MAXpd, MAXpd])
fig1.set_ylim([-MAXpd, MAXpd])
fig1.set_zlim([-MAXpd, MAXpd])
fig1 =plt.plot(np.linspace(0, eigVals[0], num=10), np.zeros(10), np.zeros(10), 'r', linewidth = 3, label = 'eigenvalue 1')
fig1 =plt.plot(np.zeros(10), np.linspace(0, eigVals[1], num=10),np.zeros(10), 'g', linewidth = 3, label = 'eigenvalue 2')
fig1 =plt.plot(np.zeros(10), np.zeros(10), np.linspace(0, eigVals[2], num=10),'b', linewidth = 3, label = 'eigenvalue 3')
plt.title('Scatter Plot of the Projected Data')
plt.legend()
plt.show
plt.savefig('/Users/hudanyun.sheng/Google Drive/Me/201808Fall/ENV6932/homework/homework1/code/3/ScatterGulfPort.png')

fig2 = plt.figure()
fig2 = plt.plot(np.arange(1, len(eigVals)+1, 1), eigVals, 'bv', markersize=5)
plt.title('Eigenvalues of the covariance matrix of Gulf Port')
plt.savefig('/Users/hudanyun.sheng/Google Drive/Me/201808Fall/ENV6932/homework/homework1/code/3/EigenvaGulfPort.png')
