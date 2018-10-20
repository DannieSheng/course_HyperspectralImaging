%%% SCRATCH: CheckMNFModes %%%

%%% INITIALIZE SIGNAL AND PARAMETERS %%%
Signal            = FakePaviaGRIm;
[NRows, NCols, B] = size(Signal);
figure(1);
subplot(2,3,1);
imagesc(FakePaviaGRIm(:, :, [PaviaRed, PaviaGreen, PaviaBlue]));

%%% INITIALIZE NOISE AND PARAMETERS %%%
RandBase          = 0.05*randn(B,B);
RandSym           = RandBase'*RandBase;
BaseCov           = 0.05*eye(103) + RandSym;
BaseNoiseVecs      = mvnrnd(zeros(B, 1), BaseCov, NRows*NCols);
BaseNoise         = reshape(BaseNoise, [NRows, NCols, B]);

%%% INITIALIZE LOOP PARAMETERS %%%
ScFac     = 0;
ScFacBase = 0.0001;
e         = exp(1);
NumLev    = 10;

%%% LOOP OVER INCREASING NOISE LEVELS %%%
for nLev = 1:NumLev
    
    %%% CALCULATE NEW NOISE AND ADD TO SIGNAL %%%
    NewNoise     = ScFac*BaseNoise;
    SigPlusNoise = Signal+ScFac*BaseNoise;
    ScFac        = 10^(nLev-1)*ScFacBase;
    TitleString  = sprintf('MNFSigPlusNoise%d', round((nLev-1)*log(ScFacBase)));
    
    %%% CORRECT SIGNAL PLUS NOISE FOR VALUES OUTSIDE OF [0,1] %%%
    %%% THIS IS NONLINEAR:  SMOOTH THRESHOLDING WITH SIGMOID  %%%
    MinX         = min(X(:));
    MaxX         = max(X(:));
    RangeX       = MaxX - MinX;
    X            = 0.7*(X -MinX)./RangeX + 0.05;
    
    %%% CALCULATE NEW COVARIANCE FOR LATER %%%
    XVecs        = reshape(X, [NRows*NCols, B]);
    Cx           = cov(XVecs);
    
    %%% RUN MNF USING DIFFERENT METHODS %%%
    Parameters.Method = 'ConstructEig';
    [Y, YVecs, Cn, ReconX, W, L] = MNFbyDGSNR(X, Parameters);
    figure(1)
    subplot(2,3,2)
    scatter3(YVecs(:, 1), YVecs(:, 2), YVecs(:, 3)); title(Parameters.Method);
    title(sprintf('%s %s', TitleString, Parameters.Method));
    subplot(2,4,5);; imagesc(mean(ReconX, 3))
    
    Parameters.Method = 'DirectEig';
    [Y, YVecs, Cn, ReconX, W, L] = MNFbyDGSNR(X, Parameters);
    figure(1)
    subplot(2,3,3)
    scatter3(YVecs(:, 1), YVecs(:, 2), YVecs(:, 3)); title(Parameters.Method);
    title(sprintf('%s %s', TitleString,Parameters.Method));
    subplot(2,3,5);; imagesc(mean(ReconX, 3))
    FileNameSavedFig = sprintf('%s%6.4f%s', 'MNFSigPlusNoise', (nLev-1)*ScFacBase, '.fig');
    figure(1)
    subplot(2,3,4)
    imagesc(X(:, :, [PaviaRed, PaviaGreen, PaviaBlue]));
    savefig(1, FileNameSavedFig, 'compact');
end

%%% OLD CODE %%%
%     SepMat            = inv(Cn)*Cx;
%     trace(SepMat);
%     fprintf('\nGenEig')
%
%     fprintf('\nL(1:3) \n');
%     fprintf(' %f ',L(1:3));
%     fprintf('\n\n');

