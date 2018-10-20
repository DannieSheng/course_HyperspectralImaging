clear 
close all
clc
dbstop if error

count = 0;

%% noise level
for level_cov = -4:2:4
    load(['/Users/hudanyun.sheng/Desktop/Project1/MNF/correct_cov', num2str(level_cov)])
    count = count+1;
    for i = 1:10
        trainCPfull(count, i) = PercCorrect{1,i}.train;
        testCPfull(count, i)  = PercCorrect{1,i}.test;
        trainCPclass(count, i) = SPercCorrect{1,i}.train;
        testCPclass(count, i)  = SPercCorrect{1,i}.test;
    end
end
mutrainCPfull  = mean(trainCPfull,2);
stdtrainCPfull = sqrt(var(trainCPfull,0,2));

mutestCPfull  = mean(testCPfull,2);
stdtestCPfull = sqrt(var(testCPfull,0,2));

mutrainCPclass  = mean(trainCPclass,2);
stdtrainCPclass = sqrt(var(trainCPclass,0,2));

mutestCPclass  = mean(testCPclass,2);
stdtestCPclass = sqrt(var(testCPclass,0,2));

figure, errorbar(-4:2:4, mutrainCPfull, stdtrainCPfull), hold on
% errorbar(-4:2:4, mutestCPfull, stdtestCPfull), hold on
errorbar(-4:2:4, mutrainCPclass, stdtrainCPclass), hold on
% errorbar(-4:2:4, mutestCPclass, stdtestCPclass)
xlabel('Noise level')
ylabel('Correct percentage of classification')
legend('Using all training data', 'Using training data for each class')
set(gca, 'FontSize', 20)

%% diagonal load level
% for level_diagload = -8:2:8
%     load(['/Users/hudanyun.sheng/Desktop/Project1/MNF/correct_DIAL', num2str(level_diagload), '.mat'])
% 	count = count+1;
%     for i = 1:10
%         trainCPfull(count, i) = PercCorrect{1,i}.train;
%         testCPfull(count, i)  = PercCorrect{1,i}.test;
%         trainCPclass(count, i) = SPercCorrect{1,i}.train;
%         testCPclass(count, i)  = SPercCorrect{1,i}.test;
%     end
% end
% mutrainCPfull  = mean(trainCPfull,2);
% stdtrainCPfull = sqrt(var(trainCPfull,0,2));
% 
% mutestCPfull  = mean(testCPfull,2);
% stdtestCPfull = sqrt(var(testCPfull,0,2));
% 
% mutrainCPclass  = mean(trainCPclass,2);
% stdtrainCPclass = sqrt(var(trainCPclass,0,2));
% 
% mutestCPclass  = mean(testCPclass,2);
% stdtestCPclass = sqrt(var(testCPclass,0,2));
% 
% figure, errorbar(-8:2:8, mutrainCPfull, stdtrainCPfull), hold on
% % errorbar(-8:2:8, mutestCPfull, stdtestCPfull), hold on
% errorbar(-8:2:8, mutrainCPclass, stdtrainCPclass), hold on
% % errorbar(-8:2:8, mutestCPclass, stdtestCPclass)
% xlabel('Diagonal loading level')
% ylabel('Correct percentage of classification')
% legend('Using all training data', 'Using training data for each class')
% set(gca, 'FontSize', 20)