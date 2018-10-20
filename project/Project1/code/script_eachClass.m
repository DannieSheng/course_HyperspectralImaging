dbstop if error
clear
close all
clc
load('Proj1Data.mat')

%% remove water bands and bad bands
% figure, plot(Proj1Wvs, Proj1Spectra(200,:), 'linewidth', 2)
wvLen   = Proj1Wvs;
spectra = Proj1Spectra;
clabel  = Proj1ClassLabels;
dataCub = Proj1Im;
% figure, plot(Proj1Wvs, Proj1Spectra, 'linewidth', 2)

idx = find((wvLen >= 1.34 & wvLen <= 1.42) | (wvLen >= 1.8 & wvLen <= 1.95) | (wvLen >= 2.45));
spectra(:, idx)  = [];
wvLen(idx)       = [];
dataCub(:,:,idx) = [];

% figure, plot(wvLen, spectra(200,:), 'linewidth', 2)
% figure, plot(wvLen, spectra, 'linewidth', 2)

%% dimensionality reduction for each class (based on label)
N = 9;
list_label = unique(clabel);
for i = 1:length(list_label)
    idx = find(clabel == list_label(i));
    X = spectra(idx, :);
    [Y, WY] = ReduceDim(X, wvLen, N);
end