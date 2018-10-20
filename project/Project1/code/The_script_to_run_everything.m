dbstop if error
clear
close all
clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% RunAll.m %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RunAll.m this script runs all the experiments for Project 1 for
% Hyperspectral Image Analysis. This code was written and edited by
% Hudanyun Sheng and Dylan Stewart for their group experiments and report.
% Within this code, the authors aim to test 4 different dimensionality
% reduction techniques for hyperspectral data on a 2 class problem to
% classify species of trees. The spectra, wavelengths and class labels are
% all contained in the file Proj1Data.mat the steps of this code are as
% follows:
%
% 1 Remove bad bands and water bands. To demonstrate the effect on the data
% we plot the mean spectra of the raw data and 2 standard deviations before
% and after removing these "bad" bands
%
% 2 Randomly select 75% of the samples of each class for training and 25%
% of each class for testing
%
% 3 Perform dimensionality reduction using PCA, MNF, Hierarchical
% Dimensionality Reduction, and Downsampling by building a transformation
% fo the ENTIRE TRAINING SET
%
% 4 Apply this transformation to the testing set
%
% 5 Train a Support Vector Machine on the TRANSFORMED TRAINING SET using
% fitPosterior(SVMModel,X,Y)
%
% 6 Calculate the percentage of TRAINING samples correctly classified
%
% 7 Calculate the percentage of TESTING samples correctly classified
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load data
load('Proj1Data.mat');

% plot the raw data with 2 standard deviations
% plot(Proj1Wvs*1000,MuSpec)
% hold on;
% plot(1000*Proj1Wvs,MuSpec+2*stdSpec);
% plot(1000*Proj1Wvs,MuSpec-2*stdSpec);
% legend('Mean Spectra','Mean +2 \sigma','Mean -2 \sigma');
% xlabel('Wavelength [nm]');
% ylabel('Magnitude');
% title('Mean and Standard Deviation of Raw Spectra')
% set(gca,'FontSize',20);

%%%%%%%%remove the bad bands%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% figure, plot(Proj1Wvs, Proj1Spectra(200,:), 'linewidth', 2)
wvLen   = Proj1Wvs;
tranIm  = permute(Proj1Im, [2,1,3]);
spectra = reshape(permute(Proj1Im, [2,1,3]), [size(Proj1Im, 1)*size(Proj1Im, 2), size(Proj1Im, 3)]);

% recover to the original shape of image
% wholeim = permute(reshape(spectra, [203,16,426]), [2,1,3]);

dataCub = Proj1Im;
% figure, plot(Proj1Wvs, Proj1Spectra, 'linewidth', 2)

idx = find((wvLen >= 1.34 & wvLen <= 1.42) | (wvLen >= 1.8 & wvLen <= 1.95) | (wvLen >= 2.45));
spectra(:, idx)  = [];
wvLen(idx)       = [];
dataCub(:,:,idx) = [];

% figure, plot(wvLen, spectra(200,:), 'linewidth', 2)
% figure, plot(wvLen, spectra, 'linewidth', 2)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% types of dimensionality reduction and experiments to run for means and
% standard deviations of performance
numExperiments = 4;
numRepititions = 100;

%%% Observations Code %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Each test we want to replicate it 100 times to be fairly certain of our
% observations
% for tests = 1:4
% for reps = 1:100

%% Train-test spliting
% Randomly select 75% of the samples of each class for training, the other
% 25% for test
percentTrain    = 0.75;
list_label      = unique(Proj1ClassLabels);
for i = 1:length(list_label)
    Nclass(i)         = sum(Proj1ClassLabels == list_label(i));
    classAll{i}       = spectra(Proj1ClassLabels == list_label(i),:);
    scrambledOrder{i} = randperm(Nclass(i) );
        
    %get indexes of training and testing
    trainInd{i}  = scrambledOrder{i}(1:percentTrain*Nclass(i));
	testInd{i}   = scrambledOrder{i}(percentTrain*Nclass(i)+1:end); 
    
    % get actual samples for each class
    trainSpec{i}  = classAll{i}(trainInd{i},:);
    testSpec{i}   = classAll{i}(testInd{i},:);
    
    % get the corresponding labels for each class
    trainLabel{i} = ones(length(trainInd{i}), 1)*list_label(i);
    testLabel{i}  = ones(length(testInd{i}), 1)*list_label(i);
end

numDim        = 40;
%%%%%%%%%%%%%%%%%%%%%%%%%%%% select sampling method %%%%%%%%%%%%%%%%%%%%%%%
%% PART 1 %%
% % Step 1: Perform dimensionality reduction using PCA, MNF, Hierarchical
% % Dimensionality Reduction, and Downsampling by building a transformation
% % on the entire training set from the TRAINING DATA and generate reduced
% % dimensionality training set T_rn
trainSpectral = [trainSpec{1}; trainSpec{2}];
testSpectral  = [testSpec{1}; testSpec{2}];
% downSamRate   = 9; % down sampling rate must be an odd number

% [T_rn, transForm]    = ReduceDim(trainSpectral, wvLen, numDesiredDim, downSamRate);
[T_rn, WY, transForm]    = ReduceDim(trainSpectral, wvLen, numDim);

% step 2: Apply the transformation to the test set generate reduced
% dimensionality test set T_en
T_en                     = ReduceDimTe(testSpectral, transForm);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PART 2 %%
% figure, imagesc(reshape(Proj1ClassLabels, [size(dataCub,1), size(dataCub, 2)]))
for i = 1:length(list_label)
    [~, SWY{i}, StransForm{i}]    = ReduceDim(trainSpec{i}, wvLen, numDim/2);
end
for i = 1:length(list_label)
    ST_rnt{i}                     = ReduceDimTe(trainSpectral, StransForm{i});
    ST_ent{i}                     = ReduceDimTe(testSpectral, StransForm{i});             
end
for i = 1:4
    ST_rn{i} = [ST_rnt{1,1}{i} ST_rnt{1,2}{i}];
    ST_en{i} = [ST_ent{1,1}{i} ST_ent{1,2}{i}];
end


% Perform dimensionality reduction using PCA, MNF, Hierarchical
% Dimensionality Reduction, and Downsampling by building a transformation
% for EACH CLASS from the TRAINING DATA and concatenate the ouputs
%
% 1 Train a Support Vector Machine on the TRANSFORMED TRAINING SET using
% fitPosterior(SVMModel,X,Y)
%
% 2 Calculate the percentage of TRAINING samples correctly classified
%
% 3 Calculate the percentage of TESTING samples correctly classified
% [T_rn, WY, transForm]    = ReduceDim(trainSpectral, wvLen, numDesiredDim/2, downSamRate*2);

%%%%%%% REDUCE DIMENSIONALITY USING ONE OF THE 4 METHODS%%%%%%%%%%%%%%%%%%%
% switch tests
%     case 1
%         % Run PCA %
%         percentVar = 0.9; %keep 90% variance
%         [trainReduced, U, Lambda] = PCAbyDG(train, percentVar);
%         Ncomps = size(trainReduced,2);
%         testReduced = projTest(test,U,Ncomps);
%     case 2
%         % Run MNF %
%         % Not sure what to do with this one %
%         %[Y, YVecs, Cn, ReconX, W] = MNFbyDG(train, MNFbyDGParameters);
%     case 3
%         % Run HDR %
%         % Not sure what to do with this one %
%         %[Clusts,ClustsFlag,ClustMems, ClustVecs,KLDMat,Hists] = ...
%         %HierarchClusterByDG(X, Wvs,HierarchClusterByDGParameters)
%     case 4
%         % Run Downsampling %
%         %[DsX, DsWvs] = SpectralDwnSmp(train, Wvs, DwnSmpRate, varargin);
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%