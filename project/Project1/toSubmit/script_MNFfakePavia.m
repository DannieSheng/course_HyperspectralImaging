clear; close all; clc
dbstop if error
load('FakePaviaThings.mat')
sz = size(FakePaviaIm);
mu = zeros(sz(1), sz(2));
R  = randn(sz(2), sz(2));
C  = R'*R;
noise = mvnrnd(mu,C);
noise = repmat(noise, [1,1,103]);
ndim  = 100;
%% FakePaviaIm
FakePaviaImNoise             = FakePaviaIm + noise;
parameters                   = MNFbyDGParameters(ndim);
[Y, YVecs, Cn, ReconX, W, L] = MNFbyDGSNR(FakePaviaImNoise, parameters);
%% FakePaviaImSq
% FakePaviaImSq    = FakePaviaImSq + noise;
figure, imagesc(mean(ReconX,3))