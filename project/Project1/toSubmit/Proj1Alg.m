dbstop if error
clear; close all; clc
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
load('ClassIm.mat'); %fake image
load('Proj1Things.mat')
load('EstNoiseFromMe'); %EstNoistFromMe: for MNF

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

dataCub          = ClassIm; %fake image
wvLen            = Proj1Wvs;
labels           = [- ones(8,203,1); ones(8,203,1)];
Proj1Labels      = reshape(permute(labels, [2,1,3]), [size(labels, 1)*size(labels, 2), size(labels, 3)]);
spectra          = reshape(permute(dataCub, [2,1,3]), [size(dataCub, 1)*size(dataCub, 2), size(dataCub, 3)]);
figure, plot(wvLen, spectra, 'linewidth', 2), xlabel('Wavelength in Micrometers'), ylabel('Estimated Reflectance')%, title('The spectra before removing bad bands')
set(gca, 'FontSize', 20)

%%%%%%%%Remove the water bands and noisy bands at the low and high wavelengths%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
idx = find((wvLen >= 1.34 & wvLen <= 1.42) | (wvLen >= 1.8 & wvLen <= 1.95) | (wvLen >= 2.45));
spectra(:, idx)         = [];
wvLen(idx)              = [];
dataCub(:,:,idx)        = [];
EstNoiseFromMe(:,:,idx) = [];
figure, plot(wvLen, spectra, 'linewidth', 2), xlabel('Wavelength in Micrometers'), ylabel('Estimated Reflectance')%, title('The spectra after removing bad bands')
set(gca, 'FontSize', 20)

%Get Noise For the entire EstNoiseMatrix
level_cov        = -5;
noise            = imnoise(EstNoiseFromMe,'gaussian',0,10^level_cov);
maxN             = max(max(max(noise)));
minN             = min(min(min(noise)));
noise            = (noise-minN)/(maxN-minN);
flag             = EstNoiseFromMe;
flag(flag<0.001) = 0;
flag(flag~=0)    = 1;
noise            = noise.*flag;
EstNoiseMask     = EstNoiseFromMe + noise.*flag;
% EstNoiseMask      = dataCub;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% types of dimensionality reduction and experiments to run for means and
% standard deviations of performance
numExperiments = 1;
numDim         = 20;
for exp = 1:numExperiments
%% Train-test spliting
% Randomly select 75% of the samples of each class for training, the other
% 25% for test

percentTrain    = 0.75;
list_label      = unique(Proj1Labels);

% train-test spliting for PCA, HDR, DownSample
for i = 1:length(list_label)
    Nclass(i)          = sum(Proj1Labels == list_label(i));
    classAll{i}        = spectra(Proj1Labels == list_label(i),:);
    scrambledOrder{i}  = randperm(Nclass(i));
        
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

% train-test spliting for MNF
MNFlabel        = [repmat(labels(1:8,:), [2,2]); repmat(labels(9:end,:), [2,2])];
% MNFlabel        = labels;
NclassC         = size(MNFlabel,2);
scrambledOrderC = randperm(NclassC);
numColTrain     = round(percentTrain*NclassC);
trainIndC       = scrambledOrderC(1:numColTrain);
testIndC        = scrambledOrderC(numColTrain+1:end);
trainIm{1}      = EstNoiseMask(1:size(EstNoiseMask,1)/2, trainIndC, :);
trainIm{2}      = EstNoiseMask(size(EstNoiseMask,1)/2+1:end, trainIndC, :);
trainLabelIm{1} = MNFlabel(1:size(EstNoiseMask,1)/2, trainIndC);
trainLabelIm{2} = MNFlabel(size(EstNoiseMask,1)/2+1:end, trainIndC);
testIm{1}       = EstNoiseMask(1:size(EstNoiseMask,1)/2, testIndC, :);
testIm{2}       = EstNoiseMask(size(EstNoiseMask,1)/2+1:end, testIndC, :);
testLabelIm{1}  = MNFlabel(1:size(EstNoiseMask,1)/2, testIndC);
testLabelIm{2}  = MNFlabel(size(EstNoiseMask,1)/2+1:end, testIndC);

%%%%%%%%%%%%%%%%%%%%%%%%%%%% select sampling method %%%%%%%%%%%%%%%%%%%%%%%
%% PART 1 %%
% % Step 1: Perform dimensionality reduction using PCA, MNF, Hierarchical
% % Dimensionality Reduction, and Downsampling by building a transformation
% % on the entire training set from the TRAINING DATA and generate reduced
% % dimensionality training set T_rn
trainSpectral = [trainSpec{1}; trainSpec{2}];
testSpectral  = [testSpec{1}; testSpec{2}];
trainLbSpc    = [trainLabel{1}; trainLabel{2}];
testLbSpc     = cat(1, testLabel{1}, testLabel{2});

trainImg   = [trainIm{1}; trainIm{2}];
testImg    = [testIm{1}; testIm{2}];
trainLbImg = [reshape(trainLabelIm{1}', [size(trainLabelIm{1},1)*size(trainLabelIm{1},2),1]);
    reshape(trainLabelIm{2}', [size(trainLabelIm{2},1)*size(trainLabelIm{2},2),1])];
testLbImg = [reshape(testLabelIm{1}', [size(testLabelIm{1},1)*size(testLabelIm{1},2),1]);
    reshape(testLabelIm{2}', [size(testLabelIm{2},1)*size(testLabelIm{2},2),1])];

% EstNoiseRows = size(EstNoiseMask,1);
% ChosenRows = randperm(EstNoiseRows);
% NoiseMaskFull = cat(1,EstNoiseMask(1:EstNoiseRows/4,1:floor(0.75*size(EstNoiseMask,2)),:),...
%    EstNoiseMask(end-(EstNoiseRows/4)+1:end,1:floor(0.75*size(EstNoiseMask,2)),:));
% NoiseMaskFull = reshape(NoiseMaskFull,[size(NoiseMaskFull,1)*size(NoiseMaskFull,2),size(NoiseMaskFull,3)]);
% NoiseMaskFull = NoiseMaskFull(randperm(size(trainSpectral,1)),:);
[T_rn, WY, transForm] = ReduceDim(trainSpectral, trainImg, wvLen, numDim);
% [T_rn, Yim, transForm, Cnb, Cn] = ReduceDim(trainSpectral, trainImg, wvLen, numDim);
% step 2: Apply the transformation to the test set generate reduced
% dimensionality test set T_en

T_en                           = ReduceDimTe(testSpectral, testImg, transForm);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:4
    trainLbl{i} = trainLbSpc;
    testLbl{i}  = testLbSpc;
end

    %% scatter plots
%     % train
% figure, scatter3(T_rn{2}(trainLbl{2} == 1,1), T_rn{2}(trainLbl{2} == 1,2), T_rn{2}(trainLbl{2} == 1,3)), hold on
% scatter3(T_rn{2}(trainLbl{2} == -1,1), T_rn{2}(trainLbl{2} == -1,2), T_rn{2}(trainLbl{2} == -1,3))
% legend('Class 1', 'Class 2')
% set(gca, 'FontSize',20)
% xlabel('MNFC1')
% ylabel('MNFC2')
% zlabel('MNFC3')
% title('First 3 components of train data after dimensionality reduction using MNF with full images')
% saveas(gcf, '/Users/hudanyun.sheng/Desktop/Project1/MNF/first3MNFtrain.jpg', 'jpg')

% figure, scatter3(T_rn{2}(trainLbl{2} == 1,end-1), T_rn{2}(trainLbl{2} == 1,end-2), T_rn{2}(trainLbl{2} == 1,end)), hold on
% scatter3(T_rn{2}(trainLbl{2} == -1,end-1), T_rn{2}(trainLbl{2} == -1,end-2), T_rn{2}(trainLbl{2} == -1,end))
% legend('Class 1', 'Class 2')
% set(gca, 'FontSize',20)
% xlabel('MNFC1')
% ylabel('MNFC2')
% zlabel('MNFC3')
% title('Last 3 components of train data after dimensionality reduction using MNF with full images')
% saveas(gcf, '/Users/hudanyun.sheng/Desktop/Project1/MNF/last3MNFtrain.jpg', 'jpg')
% 
%     % test
% figure, scatter3(T_en{2}(testLbl{2} == 1,1), T_en{2}(testLbl{2} == 1,2), T_en{2}(testLbl{2} == 1,3)), hold on
% scatter3(T_en{2}(testLbl{2} == -1,1), T_en{2}(testLbl{2} == -1,2), T_en{2}(testLbl{2} == -1,3))
% legend('Class 1', 'Class 2')
% title('First 3 components of test data after dimensionality reduction using MNF with full images')
% saveas(gcf, '/Users/hudanyun.sheng/Desktop/Project1/MNF/first3MNFtest.jpg', 'jpg')
% 
% figure, scatter3(T_en{2}(testLbl{2} == 1,end-1), T_en{2}(testLbl{2} == 1,end-2), T_en{2}(testLbl{2} == 1,end)), hold on
% scatter3(T_en{2}(testLbl{2} == -1,end-1), T_en{2}(testLbl{2} == -1,end-2), T_en{2}(testLbl{2} == -1,end))
% legend('Class 1', 'Class 2')
% title('Last 3 components of test data after dimensionality reduction using MNF with full images')
% saveas(gcf, '/Users/hudanyun.sheng/Desktop/Project1/MNF/last3MNFtest.jpg', 'jpg')
% 
%     % label for MNF (entire image)
trainLbl{2} = [reshape(trainLabelIm{1}', [size(trainLabelIm{1},1)*size(trainLabelIm{1},2),1]);
    reshape(trainLabelIm{2}', [size(trainLabelIm{2},1)*size(trainLabelIm{2},2),1])];
testLbl{2}  = [reshape(testLabelIm{1}', [size(testLabelIm{1},1)*size(testLabelIm{1},2),1]);
    reshape(testLabelIm{2}', [size(testLabelIm{2},1)*size(testLabelIm{2},2),1])];

% Randomization and Evaluation
for test = 1:length(T_rn)
    nsamps                 = length(trainLbl{test});
    scramOrder             = randperm(nsamps);
    T_rn{test}             = T_rn{test}(scramOrder,:);
    scramTrainLabel        = trainLbl{test}(scramOrder);
    scramOrderTest         = randperm(length(testLbl{test}));
    T_en{test}             = T_en{test}(scramOrderTest,:);
    scramTestLabel         = testLbl{test}(scramOrderTest);
    TrueLabels{test}.Train = scramTrainLabel;
    TrueLabels{test}.Test  = scramTestLabel;
    OutLabels{test}        = Classify(T_rn{test},T_en{test},scramTrainLabel);
    PercCorrect{exp, test}      = ScoreClassifier(TrueLabels{test},OutLabels{test});
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% PART 2 %%
% NoiseMaskTemp1 = NoiseySamps(1:size(NoiseySamps,1)/2,:,:);
% NoiseMaskTemp2 = NoiseySamps((size(NoiseySamps,1)/2)+1:end,:,:);
% NoiseMask{1} = reshape(NoiseMaskTemp1,[size(NoiseMaskTemp1,1)*size(NoiseMaskTemp1,2),...
%     size(NoiseMaskTemp1,3)]);
% NoiseMask{2} = reshape(NoiseMaskTemp2,[size(NoiseMaskTemp2,1)*size(NoiseMaskTemp2,2),...
%     size(NoiseMaskTemp2,3)]);
% figure, imagesc(reshape(Proj1ClassLabels, [size(dataCub,1), size(dataCub, 2)]))
for i = 1:length(list_label)
%     [~, SWY{i}, StransForm{i}, SCnb{i}, SCn{i}]    = ReduceDim(trainSpec{i}, trainIm{i}, wvLen, numDim/2);
    [~, SWY{i}, StransForm{i}]    = ReduceDim(trainSpec{i}, trainIm{i}, wvLen, numDim/2);
end

for i = 1:length(list_label)
    ST_rnt{i}                     = ReduceDimTe(trainSpectral, trainImg, StransForm{i});
    ST_ent{i}                     = ReduceDimTe(testSpectral, testImg, StransForm{i});             
end
for i = 1:4
    ST_rn{i} = [ST_rnt{1,1}{i} ST_rnt{1,2}{i}];
    ST_en{i} = [ST_ent{1,1}{i} ST_ent{1,2}{i}];
end

for test = 1:length(ST_rn)
    nsamps                  = length(trainLbl{test});
    scramOrder              = randperm(nsamps);
    ST_rn{test}             = ST_rn{test}(scramOrder,:);
    scramTrainLabel         = trainLbl{test}(scramOrder);
    scramOrderTest          = randperm(length(testLbl{test}));
    ST_en{test}             = ST_en{test}(scramOrderTest,:);
    scramTestLabel          = testLbl{test}(scramOrderTest);
    STrueLabels{test}.Train = scramTrainLabel;
    STrueLabels{test}.Test  = scramTestLabel;
    SOutLabels{test}        = Classify(ST_rn{test},ST_en{test},scramTrainLabel);
    SPercCorrect{exp, test} = ScoreClassifier(STrueLabels{test},SOutLabels{test});
end
end

%%%%%%%% print out the results with Named formatted arrays %%%%%%%%%
for exp = 1:numExperiments
    for test = 1:4
        trainCPf(exp, test) = PercCorrect{exp,test}.train;
        testCPf(exp, test)  = PercCorrect{exp,test}.test;
    end
end

trainCPf(numExperiments+1, :)     = mean(trainCPf(1:numExperiments, :));
trainCPf(numExperiments+2, :)     = sqrt(var(trainCPf(1:numExperiments, :)));
trainCPf                          = trainCPf*100;
trainCPf(1:numExperiments, :)     = round(trainCPf(1:numExperiments,:), 0);
trainCPf(numExperiments+1:end, :) = round(trainCPf(numExperiments+1:end, :), 1);
disp('The Training Correct Percentages for full images are:')
disp(trainCPf)

%%%%%%%% print out the results with Named formatted arrays %%%%%%%%%
testCPf(numExperiments+1, :)     = mean(testCPf(1:numExperiments, :));
testCPf(numExperiments+2, :)     = sqrt(var(testCPf(1:numExperiments, :)));
testCPf                          = testCPf*100;
testCPf(1:numExperiments, :)     = round(testCPf(1:numExperiments,:), 0);
testCPf(numExperiments+1:end, :) = round(testCPf(numExperiments+1:end, :), 1);
disp('The Test Correct Percentages for full images are:')
disp(testCPf)

for exp = 1:numExperiments
    for test = 1:4
        trainCP(exp, test) = SPercCorrect{exp,test}.train;
        testCP(exp, test)  = SPercCorrect{exp,test}.test;
    end
end
trainCP(numExperiments+1, :)     = mean(trainCP(1:numExperiments, :));
trainCP(numExperiments+2, :)     = sqrt(var(trainCP(1:numExperiments, :)));
trainCP                          = trainCP*100;
trainCP(1:numExperiments, :)     = round(trainCP(1:numExperiments,:), 0);
trainCP(numExperiments+1:end, :) = round(trainCP(numExperiments+1:end, :), 1);
disp('The Training Correct Percentages for spliting each class are:')
disp(trainCP)

testCP(numExperiments+1, :)     = mean(testCP(1:numExperiments, :));
testCP(numExperiments+2, :)     = sqrt(var(testCP(1:numExperiments, :)));
testCP                          = testCP*100;
testCP(1:numExperiments, :)     = round(testCP(1:numExperiments,:), 0);
testCP(numExperiments+1:end, :) = round(testCP(numExperiments+1:end, :), 1);
disp('The Test Correct Percentages for spliting each class are:')
disp(testCP)

% save(['/Users/hudanyun.sheng/Desktop/Project1/errorBar/data_with_dim', num2str(numDim), '.mat'], 'trainCPf', 'testCPf', 'trainCP', 'testCP')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%