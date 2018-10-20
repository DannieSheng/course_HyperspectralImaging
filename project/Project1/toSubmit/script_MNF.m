dbstop if error
clear; close all; clc
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% the script to see parameters for MNF
% load data
load('ClassIm.mat'); %fake image
load('Proj1Things.mat')
load('EstNoiseFromMe'); %EstNoistFromMe: for MNF

dataCub          = ClassIm; %fake image
wvLen            = Proj1Wvs;
labels           = [- ones(8,203,1); ones(8,203,1)];
Proj1Labels      = reshape(permute(labels, [2,1,3]), [size(labels, 1)*size(labels, 2), size(labels, 3)]);
spectra          = reshape(permute(dataCub, [2,1,3]), [size(dataCub, 1)*size(dataCub, 2), size(dataCub, 3)]);

%%%%%%%%Remove the water bands and noisy bands at the low and high wavelengths%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

idx = find((wvLen >= 1.34 & wvLen <= 1.42) | (wvLen >= 1.8 & wvLen <= 1.95) | (wvLen >= 2.45));
spectra(:, idx)         = [];
wvLen(idx)              = [];
dataCub(:,:,idx)        = [];
EstNoiseFromMe(:,:,idx) = [];

numExperiments = 10;
numDim         = 20;

%% different covariance matrix
%Get Noise For the entire EstNoiseMatrix
B               = size(EstNoiseFromMe,3);
A               = rand(B, B);
C               = A'*A;
mu              = zeros(B,1);
for level_cov = -4:2:4
    cov_m        = C*10^level_cov;
    noise        = imnoise(EstNoiseFromMe,'gaussian',0,10^level_cov);
    EstNoiseMask = EstNoiseFromMe + reshape(noise, size(EstNoiseFromMe));
for exp = 1:numExperiments
%% Train-test spliting
% Randomly select 75% of the samples of each class for training, the other
% 25% for test

        percentTrain    = 0.75;
        list_label      = unique(Proj1Labels);

        % train-test spliting for MNF
        MNFlabel        = [repmat(labels(1:8,:), [2,2]); repmat(labels(9:end,:), [2,2])];
        NclassC         = size(MNFlabel,2);
        scrambledOrderC = randperm(NclassC);
        numColTrain     = round(percentTrain*NclassC);
        trainIndC       = scrambledOrderC(1:numColTrain);
        testIndC        = scrambledOrderC(numColTrain+1:end);
        trainIm{1}      = EstNoiseMask(1:16, trainIndC, :);
        trainIm{2}      = EstNoiseMask(17:end, trainIndC, :);
        trainLabelIm{1} = MNFlabel(1:16, trainIndC);
        trainLabelIm{2} = MNFlabel(17:end, trainIndC);
        testIm{1}       = EstNoiseMask(1:16, testIndC, :);
        testIm{2}       = EstNoiseMask(17:end, testIndC, :);
        testLabelIm{1}  = MNFlabel(1:16, testIndC);
        testLabelIm{2}  = MNFlabel(17:end, testIndC);


%%%%%%%%%%%%%%%%%%%%%%%%%%%% select sampling method %%%%%%%%%%%%%%%%%%%%%%%
    %% PART 1 %%

        trainImg   = [trainIm{1}; trainIm{2}];
        testImg    = [testIm{1}; testIm{2}];

        mnfParameters                      = MNFbyDGParameters();
        mnfParameters.NComps               = numDim;
        [T_rnim, T_rn, Cnb, Cn, transform] = MNFbyDGSNR(trainImg, mnfParameters);

% step 2: Apply the transformation to the test set generate reduced
% dimensionality test set T_en
        Sx        = size(trainImg);
        NRows     = Sx(1);
        NCols     = Sx(2);
        ZEROMEAN  = transform.parameters.ZEROMEAN;
        XVecs     = reshape(permute(trainImg, [2,1,3]), [NRows*NCols, B]);
        if(ZEROMEAN)
            XmuBig = repmat(transform.parameters.Xmu, [NRows*NCols, 1]);
            XVecs  = XVecs - XmuBig;
        end

        T_en      = fliplr((transform.W(:, (end-transform.parameters.NComps+1:end))'*XVecs')');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        trainLbl = [reshape(trainLabelIm{1}', [size(trainLabelIm{1},1)*size(trainLabelIm{1},2),1]);
    reshape(trainLabelIm{2}', [size(trainLabelIm{2},1)*size(trainLabelIm{2},2),1])];
        testLbl  = [reshape(testLabelIm{1}', [size(testLabelIm{1},1)*size(testLabelIm{1},2),1]);
    reshape(testLabelIm{2}', [size(testLabelIm{2},1)*size(testLabelIm{2},2),1])];
    
    figure, scatter3(T_rn(trainLbl == 1, 1), T_rn(trainLbl == 1, 2), T_rn(trainLbl == 1, 3)), hold on
    scatter3(T_rn(trainLbl == -1, 1), T_rn(trainLbl == -1, 2), T_rn(trainLbl == -1, 3))
    set(gcf,'outerposition',get(0,'screensize'));
    title(['Scatter plot of train data after dimensionality reduction using full image with MNF when level of covariance matrix is ', num2str(level_cov)])
    legend('Positive class', 'Negative class')
    set(gca, 'FontSize', 20)
    saveas(gcf, ['/Users/hudanyun.sheng/Desktop/Project1/MNF/scatterMNFfull_cov', num2str(level_cov), 'train.jpg'], 'jpg')
    
	figure, scatter3(T_en(trainLbl == 1, 1), T_en(trainLbl == 1, 2), T_en(trainLbl == 1, 3)), hold on
    scatter3(T_en(testLbl == -1, 1), T_en(testLbl == -1, 2), T_en(testLbl == -1, 3))
    set(gcf,'outerposition',get(0,'screensize'));
    title(['Scatter plot of test data after dimensionality reduction using full image with MNF when level of covariance matrix is ', num2str(level_cov)])
    legend('Positive class', 'Negative class')
    set(gca, 'FontSize', 20)
    saveas(gcf, ['/Users/hudanyun.sheng/Desktop/Project1/MNF/scatterMNFfull_cov', num2str(level_cov), 'test.jpg'], 'jpg')
    
    %Randomization and Evaluation
        nsamps                 = length(trainLbl);
        scramOrder             = randperm(nsamps);
        T_rn                   = T_rn(scramOrder,:);
        scramTrainLabel        = trainLbl(scramOrder);
        scramOrderTest         = randperm(length(testLbl));
        T_en                   = T_en(scramOrderTest,:);
        scramTestLabel         = testLbl(scramOrderTest);
        TrueLabels.Train       = scramTrainLabel;
        TrueLabels.Test        = scramTestLabel;
        OutLabels              = Classify(T_rn,T_en,scramTrainLabel);
        PercCorrect{exp}       = ScoreClassifier(TrueLabels,OutLabels);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% PART 2 %%

        for i = 1:length(list_label)
            mnfParameters                      = MNFbyDGParameters();
            mnfParameters.NComps               = numDim/2;
            [ST_rnim, ~, SCnb, SCn, Stransform] = MNFbyDGSNR(trainImg, mnfParameters);
        end

        for i = 1:length(list_label)
%     ST_rnt{i}                     = ReduceDimTe(trainSpectral, trainImg, StransForm{i});   
            Sx        = size(trainImg);
            NRows     = Sx(1);
            NCols     = Sx(2);
            ZEROMEAN  = transform.parameters.ZEROMEAN;
            XVecs     = reshape(permute(trainImg, [2,1,3]), [NRows*NCols, B]);
            if(ZEROMEAN)
                XmuBig = repmat(transform.parameters.Xmu, [NRows*NCols, 1]);
                XVecs  = XVecs - XmuBig;
            end
            ST_rnt{i}      = fliplr((transform.W(:, (end-transform.parameters.NComps+1:end))'*XVecs')');
    
% 	ST_ent{i}                     = ReduceDimTe(testSpectral, testImg, StransForm{i}); 
            Sx        = size(testImg);
            NRows     = Sx(1);
            NCols     = Sx(2);
            XVecs     = reshape(permute(testImg, [2,1,3]), [NRows*NCols, B]);
            if(ZEROMEAN)
                XmuBig = repmat(transform.parameters.Xmu, [NRows*NCols, 1]);
                XVecs  = XVecs - XmuBig;
            end
            ST_ent{i}      = fliplr((transform.W(:, (end-transform.parameters.NComps+1:end))'*XVecs')');
        end

        ST_rn = [ST_rnt{1,1} ST_rnt{1,2}];
        ST_en = [ST_ent{1,1} ST_ent{1,2}];
        
	figure, scatter3(ST_rn(trainLbl == 1, 1), ST_rn(trainLbl == 1, 2), ST_rn(trainLbl == 1, 3)), hold on
    scatter3(ST_rn(trainLbl == -1, 1), ST_rn(trainLbl == -1, 2), ST_rn(trainLbl == -1, 3))
    set(gcf,'outerposition',get(0,'screensize'));
    title(['Scatter plot of train data after dimensionality reduction using each class with MNF when level of covariance matrix is ', num2str(level_cov)])
    legend('Positive class', 'Negative class')
    saveas(gcf, ['/Users/hudanyun.sheng/Desktop/Project1/MNF/scatterMNFclass_cov', num2str(level_cov), 'train.jpg'], 'jpg')
    
	figure, scatter3(ST_en(testLbl == 1, 1), ST_en(testLbl == 1, 2), ST_en(testLbl == 1, 3)), hold on
    scatter3(ST_en(testLbl == -1, 1), ST_en(testLbl == -1, 2), ST_en(testLbl == -1, 3))
    set(gcf,'outerposition',get(0,'screensize'));
    title(['Scatter plot of test data after dimensionality reduction using each class with MNF when level of covariance matrix is ', num2str(level_cov)])
    legend('Positive class', 'Negative class')
    saveas(gcf, ['/Users/hudanyun.sheng/Desktop/Project1/MNF/scatterMNFclass_cov', num2str(level_cov), 'test.jpg'], 'jpg')

        nsamps            = length(trainLbl);
        scramOrder        = randperm(nsamps);
        ST_rn             = ST_rn(scramOrder,:);
        scramTrainLabel   = trainLbl(scramOrder);
        scramOrderTest    = randperm(length(testLbl));
        ST_en             = ST_en(scramOrderTest,:);
        scramTestLabel    = testLbl(scramOrderTest);
        STrueLabels.Train = scramTrainLabel;
        STrueLabels.Test  = scramTestLabel;
        SOutLabels        = Classify(ST_rn, ST_en,scramTrainLabel);
        SPercCorrect{exp} = ScoreClassifier(STrueLabels,SOutLabels);
        close all
    end
    close all
    save(['/Users/hudanyun.sheng/Desktop/Project1/MNF/correct_cov' num2str(level_cov) '.mat'], 'PercCorrect', 'SPercCorrect')
end

%% different diagonal loading
% for level_diagload = -8:2:8
%     EstNoiseMask = EstNoiseFromMe;
%     for exp = 1:numExperiments
%     %% Train-test spliting
%     % Randomly select 75% of the samples of each class for training, the other
%     % 25% for test
%         percentTrain    = 0.75;
%         list_label      = unique(Proj1Labels);
% 
%         % train-test spliting for MNF
%         MNFlabel        = [repmat(labels(1:8,:), [2,2]); repmat(labels(9:end,:), [2,2])];
%         NclassC         = size(MNFlabel,2);
%         scrambledOrderC = randperm(NclassC);
%         numColTrain     = round(percentTrain*NclassC);
%         trainIndC       = scrambledOrderC(1:numColTrain);
%         testIndC        = scrambledOrderC(numColTrain+1:end);
%         trainIm{1}      = EstNoiseMask(1:16, trainIndC, :);
%         trainIm{2}      = EstNoiseMask(17:end, trainIndC, :);
%         trainLabelIm{1} = MNFlabel(1:16, trainIndC);
%         trainLabelIm{2} = MNFlabel(17:end, trainIndC);
%         testIm{1}       = EstNoiseMask(1:16, testIndC, :);
%         testIm{2}       = EstNoiseMask(17:end, testIndC, :);
%         testLabelIm{1}  = MNFlabel(1:16, testIndC);
%         testLabelIm{2}  = MNFlabel(17:end, testIndC);
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%% select sampling method %%%%%%%%%%%%%%%%%%%%%%%
%     %% PART 1 %%
% 
%         trainImg   = [trainIm{1}; trainIm{2}];
%         testImg    = [testIm{1}; testIm{2}];
% 
%         mnfParameters                      = MNFbyDGParameters();
%         mnfParameters.NComps               = numDim;
%         mnfParameters.DIAGLOAD             = 0.5*10^level_diagload;
%         [T_rnim, T_rn, Cnb, Cn, transform] = MNFbyDGSNR(trainImg, mnfParameters);
% 
% % step 2: Apply the transformation to the test set generate reduced
% % dimensionality test set T_en
%         Sx        = size(trainImg);
%         NRows     = Sx(1);
%         NCols     = Sx(2);
%         ZEROMEAN  = transform.parameters.ZEROMEAN;
%         XVecs     = reshape(permute(trainImg, [2,1,3]), [NRows*NCols, B]);
%         if(ZEROMEAN)
%             XmuBig = repmat(transform.parameters.Xmu, [NRows*NCols, 1]);
%             XVecs  = XVecs - XmuBig;
%         end
% 
%         T_en      = fliplr((transform.W(:, (end-transform.parameters.NComps+1:end))'*XVecs')');
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%         trainLbl = [reshape(trainLabelIm{1}', [size(trainLabelIm{1},1)*size(trainLabelIm{1},2),1]);
%     reshape(trainLabelIm{2}', [size(trainLabelIm{2},1)*size(trainLabelIm{2},2),1])];
%         testLbl  = [reshape(testLabelIm{1}', [size(testLabelIm{1},1)*size(testLabelIm{1},2),1]);
%     reshape(testLabelIm{2}', [size(testLabelIm{2},1)*size(testLabelIm{2},2),1])];
%     
% %     figure, scatter3(T_rn(trainLbl == 1, 1), T_rn(trainLbl == 1, 2), T_rn(trainLbl == 1, 3)), hold on
% %     scatter3(T_rn(trainLbl == -1, 1), T_rn(trainLbl == -1, 2), T_rn(trainLbl == -1, 3))
% %     set(gcf,'outerposition',get(0,'screensize'));
% %     title(['Scatter plot of train data after dimensionality reduction using full image with MNF when level of diagonal load is ', num2str(level_diagload)])
% %     legend('Positive class', 'Negative class')
% %     saveas(gcf, ['/Users/hudanyun.sheng/Desktop/Project1/MNF/scatterMNFfull_DIAG', num2str(level_diagload), 'train.jpg'], 'jpg')
%     
% % 	figure, scatter3(T_en(trainLbl == 1, 1), T_en(trainLbl == 1, 2), T_en(trainLbl == 1, 3)), hold on
% %     scatter3(T_en(testLbl == -1, 1), T_en(testLbl == -1, 2), T_en(testLbl == -1, 3))
% %     set(gcf,'outerposition',get(0,'screensize'));
% %     title(['Scatter plot of test data after dimensionality reduction using full image with MNF when level of diagonal load is ', num2str(level_diagload)])
% %     legend('Positive class', 'Negative class')
% %     saveas(gcf, ['/Users/hudanyun.sheng/Desktop/Project1/MNF/scatterMNFfull_DIAG', num2str(level_diagload), 'test.jpg'], 'jpg')
%     
%     %Randomization and Evaluation
%         nsamps                 = length(trainLbl);
%         scramOrder             = randperm(nsamps);
%         T_rn                   = T_rn(scramOrder,:);
%         scramTrainLabel        = trainLbl(scramOrder);
%         scramOrderTest         = randperm(length(testLbl));
%         T_en                   = T_en(scramOrderTest,:);
%         scramTestLabel         = testLbl(scramOrderTest);
%         TrueLabels.Train       = scramTrainLabel;
%         TrueLabels.Test        = scramTestLabel;
%         OutLabels              = Classify(T_rn,T_en,scramTrainLabel);
%         PercCorrect{exp}       = ScoreClassifier(TrueLabels,OutLabels);
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% %% PART 2 %%
% 
%         for i = 1:length(list_label)
%             mnfParameters                      = MNFbyDGParameters();
%             mnfParameters.NComps               = numDim/2;
%             mnfParameters.DIAGLOAD             = 0.5*10^level_diagload;
%             [ST_rnim, ST_rn, SCnb, SCn, Stransform] = MNFbyDGSNR(trainImg, mnfParameters);
%         end
% 
%         for i = 1:length(list_label)
% %     ST_rnt{i}                     = ReduceDimTe(trainSpectral, trainImg, StransForm{i});   
%             Sx        = size(trainImg);
%             NRows     = Sx(1);
%             NCols     = Sx(2);
%             ZEROMEAN  = transform.parameters.ZEROMEAN;
%             XVecs     = reshape(permute(trainImg, [2,1,3]), [NRows*NCols, B]);
%             if(ZEROMEAN)
%                 XmuBig = repmat(transform.parameters.Xmu, [NRows*NCols, 1]);
%                 XVecs  = XVecs - XmuBig;
%             end
%             ST_rnt{i}      = fliplr((transform.W(:, (end-transform.parameters.NComps+1:end))'*XVecs')');
%     
% % 	ST_ent{i}                     = ReduceDimTe(testSpectral, testImg, StransForm{i}); 
%             Sx        = size(testImg);
%             NRows     = Sx(1);
%             NCols     = Sx(2);
%             XVecs     = reshape(permute(testImg, [2,1,3]), [NRows*NCols, B]);
%             if(ZEROMEAN)
%                 XmuBig = repmat(transform.parameters.Xmu, [NRows*NCols, 1]);
%                 XVecs  = XVecs - XmuBig;
%             end
%             ST_ent{i}      = fliplr((transform.W(:, (end-transform.parameters.NComps+1:end))'*XVecs')');
%         end
% 
%         ST_rn = [ST_rnt{1,1} ST_rnt{1,2}];
%         ST_en = [ST_ent{1,1} ST_ent{1,2}];
%         
% % 	figure, scatter3(ST_rn(trainLbl == 1, 1), ST_rn(trainLbl == 1, 2), ST_rn(trainLbl == 1, 3)), hold on
% %     scatter3(ST_rn(trainLbl == -1, 1), ST_rn(trainLbl == -1, 2), ST_rn(trainLbl == -1, 3))
% %     set(gcf,'outerposition',get(0,'screensize'));
% %     title(['Scatter plot of train data after dimensionality reduction using each class with MNF when level of diagonal load is ', num2str(level_diagload)])
% %     legend('Positive class', 'Negative class')
% %     saveas(gcf, ['/Users/hudanyun.sheng/Desktop/Project1/MNF/scatterMNFclass_DIAG', num2str(level_diagload), 'train.jpg'], 'jpg')
%     
% % 	figure, scatter3(ST_en(testLbl == 1, 1), ST_en(testLbl == 1, 2), ST_en(testLbl == 1, 3)), hold on
% %     scatter3(ST_en(testLbl == -1, 1), ST_en(testLbl == -1, 2), ST_en(testLbl == -1, 3))
% %     set(gcf,'outerposition',get(0,'screensize'));
% %     title(['Scatter plot of test data after dimensionality reduction using each class with MNF when level of diagonal load is ', num2str(level_diagload)])
% %     legend('Positive class', 'Negative class')
% %     saveas(gcf, ['/Users/hudanyun.sheng/Desktop/Project1/MNF/scatterMNFclass_DIAG', num2str(level_diagload), 'test.jpg'], 'jpg')
% 
%         nsamps            = length(trainLbl);
%         scramOrder        = randperm(nsamps);
%         ST_rn             = ST_rn(scramOrder,:);
%         scramTrainLabel   = trainLbl(scramOrder);
%         scramOrderTest    = randperm(length(testLbl));
%         ST_en             = ST_en(scramOrderTest,:);
%         scramTestLabel    = testLbl(scramOrderTest);
%         STrueLabels.Train = scramTrainLabel;
%         STrueLabels.Test  = scramTestLabel;
%         SOutLabels        = Classify(ST_rn, ST_en,scramTrainLabel);
%         SPercCorrect{exp} = ScoreClassifier(STrueLabels,SOutLabels);
%     end
%     save(['/Users/hudanyun.sheng/Desktop/Project1/MNF/correct_DIAL' num2str(level_diagload) '.mat'], 'PercCorrect', 'SPercCorrect')
% end