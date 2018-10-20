dbstop if error
clear; close all; clc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load data
load('FakePaviaThings.mat'); %FakePaviaIm, FakepaviaImSq, PaviaWavelengths, PaviaColors

sz = size(FakePaviaIm);
B = sz(3);

figure, imagesc(mean(FakePaviaIm,3))
figure, imagesc(mean(FakePaviaIm,3))

gr = reshape(FakePaviaIm(:, [1:8, 17:24], :), [sz(1)*sz(2)/2, B]);
rd = reshape(FakePaviaIm(:, [9:16, 25:32], :), [sz(1)*sz(2)/2, B]);
figure, scatter3(gr(:,1), gr(:,2), gr(:,3), 'g'), hold on
scatter3(rd(:,1), rd(:,2), rd(:,3), 'r')
legend('Green pixels', 'Red pixels')
% title('Scatter plot of first 3 dimensions before dimensionality reduction of Pavia data')
set(gca, 'FontSize', 20)
set(gcf,'outerposition',get(0,'screensize'));
saveas(gcf, '/Users/hudanyun.sheng/Desktop/Project1/pavia/scatterPavia.jpg')

grSq = reshape(FakePaviaImSq(:, [1:8, 17:24], :), [sz(1)*sz(2)/2, B]);
rdSq = reshape(FakePaviaImSq(:, [9:16, 25:32], :), [sz(1)*sz(2)/2, B]);
% figure, scatter3(grSq(:,1), grSq(:,2), grSq(:,3), 'g'), hold on
% scatter3(rdSq(:,1), rdSq(:,2), rdSq(:,3), 'r')
% legend('Green pixels', 'Red pixels')
% % title('Scatter plot of first 3 dimensions before dimensionality reduction of squeezed Pavia data')
% set(gca, 'FontSize', 20)
% set(gcf,'outerposition',get(0,'screensize'));
% saveas(gcf, '/Users/hudanyun.sheng/Desktop/Project1/pavia/scatterPaviaSqueezed.jpg')

A        = rand(B,B);
cov_mat  = A'*A;
mu       = zeros(B,1);
noise    = mvnrnd(mu, cov_mat, sz(1)*sz(2));
minN     = min(min(noise));
maxN     = max(max(noise));
noise    = (noise-minN)/(maxN-minN);
noise    = reshape(noise, size(FakePaviaIm));

%%%%%%% Experiment with different noise levels %%%%%%%%%
for level_noise = -4:2:4
    mnfParameters = MNFbyDGParameters();
    mnfParameters.NComps = 4;
    mnfParameters.DIAGLOAD = 0.5;
    X             = FakePaviaIm + noise*10^level_noise;
	figure, imagesc(mean(X,3)), title(['Mean over wavelengths of Pavia image with noise level 10e', num2str(level_noise)])
    saveas(gcf, ['/Users/hudanyun.sheng/Desktop/Project1/pavia/mean', num2str(level_noise),'.jpg'])
    
	[Y, YVecs, Cnb, Cn, transform]           = MNFbyDGSNR(X, mnfParameters);
%     figure, imagesc(mean(Y,3)), title(['MNF dimensionality reduction result of Pavia image with noise level 10e', num2str(level_noise)])
% 	set(gca, 'FontSize', 20)
%     set(gcf,'outerposition',get(0,'screensize'));
%     saveas(gcf, ['/Users/hudanyun.sheng/Desktop/Project1/pavia/mnf', num2str(level_noise),'.jpg'])
    
    szY = size(Y);
    Ygr = reshape(Y(:, [1:8, 17:24], :), [szY(1)*szY(2)/2, szY(3)]);
    Yrd = reshape(Y(:, [9:16, 25:32], :), [szY(1)*szY(2)/2, szY(3)]);
    figure, scatter3(Ygr(:,1), Ygr(:,2), Ygr(:,3), 'g'), hold on
    scatter3(Yrd(:,1), Yrd(:,2), Yrd(:,3), 'r')
    legend('Green pixels', 'Red pixels')
    title(['Scatter plot of first 3 dimensions after dimensionality reduction of Pavia data with noise level 10e', num2str(level_noise)])
    set(gca, 'FontSize', 20)
    set(gcf,'outerposition',get(0,'screensize'));
    saveas(gcf, ['/Users/hudanyun.sheng/Desktop/Project1/pavia/Noisescattermnf', num2str(level_noise), '.jpg'])

    
    XSq           = FakePaviaImSq + noise*10^level_noise; 
% 	figure, imagesc(mean(XSq,3)), title(['Mean over wavelengths of squeezed Pavia image with noise level 10e', num2str(level_noise)])
%     saveas(gcf, ['meansqueezed', num2str(level_noise),'.jpg'])
      
    [YSq, YVecsSq, CnbSq, CnSq, transformSq] = MNFbyDGSNR(XSq, mnfParameters);
% 	figure, imagesc(mean(YSq,3)), title(['MNF dimensionality reduction result of squeezed Pavia image with noise level 10e', num2str(level_noise)])
% 	set(gca, 'FontSize', 20)
%     set(gcf,'outerposition',get(0,'screensize'));
%     saveas(gcf, ['/Users/hudanyun.sheng/Desktop/Project1/pavia/mnfsqueezed', num2str(level_noise),'.jpg'])
    
    szYSq = size(YSq);
    YSqgr = reshape(YSq(:, [1:8, 17:24], :), [szYSq(1)*szYSq(2)/2, szYSq(3)]);
    YSqrd = reshape(YSq(:, [9:16, 25:32], :), [szYSq(1)*szYSq(2)/2, szYSq(3)]);
    figure, scatter3(YSqgr(:,1), YSqgr(:,2), YSqgr(:,3), 'g'), hold on
    scatter3(YSqrd(:,1), YSqrd(:,2), YSqrd(:,3), 'r')
    legend('Green pixels', 'Red pixels')
    title(['Scatter plot of first 3 dimensions after dimensionality reduction of squeezed Pavia data with noise level 10e', num2str(level_noise)])
    set(gca, 'FontSize', 20)
    set(gcf,'outerposition',get(0,'screensize'));
    saveas(gcf, ['/Users/hudanyun.sheng/Desktop/Project1/pavia/Noisescattermnfsqueezed', num2str(level_noise), '.jpg'])
end

close all
%%%%%%%% Experiment with different diagonal load levels %%%%%%%%%%
for level_diagload = -8:2:8
    mnfParameters = MNFbyDGParameters();
    mufParameters.NComps   = 4;
    mnfParameters.DIAGLOAD = 0.5*10^level_diagload;
    
    [Y, YVecs, Cnb, Cn, transform]           = MNFbyDGSNR(X, mnfParameters);
	figure, imagesc(mean(Y,3)), title(['MNF dimensionality reduction result of Pavia image with diagonal load level 10e', num2str(level_diagload)])
	set(gca, 'FontSize', 20)
    set(gcf,'outerposition',get(0,'screensize'));
    saveas(gcf, ['/Users/hudanyun.sheng/Desktop/Project1/pavia/mnfdiag', num2str(level_diagload),'.jpg'])
    
    szY = size(Y);
    Ygr = reshape(Y(:, [1:8, 17:24], :), [szY(1)*szY(2)/2, szY(3)]);
    Yrd = reshape(Y(:, [9:16, 25:32], :), [szY(1)*szY(2)/2, szY(3)]);
    figure, scatter3(Ygr(:,1), Ygr(:,2), Ygr(:,3), 'g'), hold on
    scatter3(Yrd(:,1), Yrd(:,2), Yrd(:,3), 'r')
    legend('Green pixels', 'Red pixels')
    title(['Scatter plot of first 3 dimensions after dimensionality reduction of Pavia data with diagonal load level 10e', num2str(level_diagload)])
    set(gca, 'FontSize', 20)
    set(gcf,'outerposition',get(0,'screensize'));
    saveas(gcf, ['/Users/hudanyun.sheng/Desktop/Project1/pavia/scattermnf', num2str(level_diagload), '.jpg'])
    
    
	[YSq, YVecsSq, CnbSq, CnSq, transformSq] = MNFbyDGSNR(XSq, mnfParameters);
	figure, imagesc(mean(YSq,3)), title(['MNF dimensionality reduction result of squeezed Pavia image with diagonal load level 10e', num2str(level_diagload)])
	set(gca, 'FontSize', 20)
    set(gcf,'outerposition',get(0,'screensize'));   
    saveas(gcf, ['/Users/hudanyun.sheng/Desktop/Project1/pavia/mnfdiagsqueezed', num2str(level_diagload),'.jpg'])
    
	szYSq = size(YSq);
    YSqgr = reshape(YSq(:, [1:8, 17:24], :), [szYSq(1)*szYSq(2)/2, szYSq(3)]);
    YSqrd = reshape(YSq(:, [9:16, 25:32], :), [szYSq(1)*szYSq(2)/2, szYSq(3)]);
    figure, scatter3(YSqgr(:,1), YSqgr(:,2), YSqgr(:,3), 'g'), hold on
    scatter3(YSqrd(:,1), YSqrd(:,2), YSqrd(:,3), 'r')
    legend('Green pixels', 'Red pixels')
    title(['Scatter plot of first 3 dimensions after dimensionality reduction of squeezed Pavia data with diagonal load level 10e', num2str(level_diagload)])
    set(gca, 'FontSize', 20)
    set(gcf,'outerposition',get(0,'screensize'));
    saveas(gcf, ['/Users/hudanyun.sheng/Desktop/Project1/pavia/scattermnfsqueezed', num2str(level_diagload), '.jpg'])
end