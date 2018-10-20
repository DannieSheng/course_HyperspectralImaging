clear
close all
clc
list_dim = [2:2:20 30:30:180];
mu_fullTrain   = [];
mu_fullTest    = [];
mu_classTrain  = [];
mu_classTest   = [];
std_fullTrain  = [];
std_fullTest   = [];
std_classTrain = [];
std_classTest  = [];
for i = 1:length(list_dim)
    numDim = list_dim(i);
    load(['data_with_dim', num2str(numDim), '.mat'])
    mu_fullTrain  = [mu_fullTrain trainCPf(11,:)'];
    mu_fullTest   = [mu_fullTest testCPf(11,:)'];
    mu_classTrain = [mu_classTrain trainCP(11,:)'];
    mu_classTest  = [mu_classTest testCP(11,:)'];
    std_fullTrain  = [std_fullTrain trainCPf(12,:)'];
    std_fullTest   = [std_fullTest testCPf(12,:)'];
    std_classTrain = [std_classTrain testCP(12,:)'];
    std_classTest  = [std_classTest testCP(12,:)'];
end

    % full train
figure, errorbar(list_dim, mu_fullTrain(1,:), 2*std_fullTrain(1,:)), hold on
errorbar(list_dim, mu_fullTrain(2,:), 2*std_fullTrain(2,:)), hold on
errorbar(list_dim, mu_fullTrain(3,:), 2*std_fullTrain(3,:)), hold on
errorbar(list_dim, mu_fullTrain(4,:), 2*std_fullTrain(4,:))
legend('PCA', 'MNF', 'HDR', 'DownSampling', 'Location','southeast')
xlabel('Number of dimensions kept')
ylabel('Classification correct percentages(%)')
title('Error bar for train data using full image')
set(gca, 'FontSize',20)
set(gcf,'outerposition',get(0,'screensize'));
saveas(gcf, 'fullTrain.jpg', 'jpg')

    % full test
figure, errorbar(list_dim, mu_fullTest(1,:), 2*std_fullTest(1,:)), hold on
errorbar(list_dim, mu_fullTest(2,:), 2*std_fullTest(2,:)), hold on
errorbar(list_dim, mu_fullTest(3,:), 2*std_fullTest(3,:)), hold on
errorbar(list_dim, mu_fullTest(4,:), 2*std_fullTest(4,:))
legend('PCA', 'MNF', 'HDR', 'DownSampling', 'Location', 'southeast')
xlabel('Number of dimensions kept')
ylabel('Classification correct percentages(%)')
title('Error bar for test data using full image')
set(gca, 'FontSize',20)
set(gcf,'outerposition',get(0,'screensize'));
saveas(gcf, 'fullTest.jpg', 'jpg')

    % class train
figure, errorbar(list_dim, mu_classTrain(1,:), 2*std_classTrain(1,:)), hold on
errorbar(list_dim, mu_classTrain(2,:), 2*std_classTrain(2,:)), hold on
errorbar(list_dim, mu_classTrain(3,:), 2*std_classTrain(3,:)), hold on
errorbar(list_dim, mu_classTrain(4,:), 2*std_classTrain(4,:))
legend('PCA', 'MNF', 'HDR', 'DownSampling', 'Location', 'southeast')
xlabel('Number of dimensions kept')
ylabel('Classification correct percentages(%)')
title('Error bar for train data using image segment from each class')
set(gca, 'FontSize',20)
set(gcf,'outerposition',get(0,'screensize'));
saveas(gcf, 'classTrain.jpg', 'jpg')

    % class test
figure, errorbar(list_dim, mu_classTest(1,:), std_classTest(1,:)*2), hold on
errorbar(list_dim, mu_classTest(2,:), std_classTest(2,:)*2), hold on
errorbar(list_dim, mu_classTest(3,:), std_classTest(3,:)*2), hold on
errorbar(list_dim, mu_classTest(4,:), std_classTest(4,:)*2)
legend('PCA', 'MNF', 'HDR', 'DownSampling', 'Location', 'southeast')
xlabel('Number of dimensions kept')
ylabel('Classification correct percentages(%)')
title('Error bar for test data using image segment from each class')
set(gca, 'FontSize',20)
set(gcf,'outerposition',get(0,'screensize'));
saveas(gcf, 'classTest.jpg', 'jpg')