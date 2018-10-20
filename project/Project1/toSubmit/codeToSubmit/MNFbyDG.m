function  [Y, YVecs, transform] = MNFbyDG(X, Parameters)
%function [Y, YVecs, Cn, ReconX, W] = MNFbyDG(X, NComps, NoiseMask)
%
%%% X is an NRows x NCols X B Spectral Data Cube
%%%     where
%%%        B = Number of Bands
%%%        N = NRows x NCols = Number of Spectra
%%%
%%% Parameters COME FROM THE FILE MNFbyDGParameters.m
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% AUTHOR: Darth Gader %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
%%% SET PARAMETER VARIABLES. SEE THE FILE MNFbyDGParameters.m  %%%
NComps    = Parameters.NComps;
NoiseMask = Parameters.NoiseMask;
Method    = Parameters.Method;
ZEROMEAN  = Parameters.ZEROMEAN;
DIAGLOAD  = Parameters.DIAGLOAD;
%%
%%% INITIALIZE SIZES %%%
Sx    = size(X);
NRows = Sx(1);
NCols = Sx(2);
B     = Sx(3);
N     = NRows*NCols;
%%
%%% CALCULATE  NOISE COVARIANCE %%%
SubNbrs = zeros(size(X));
for b = 1:B
    XSlice           = squeeze(X(:, :, b));
    SubNbrs(:, :, b) = conv2(XSlice, NoiseMask, 'same');
end
SubNbrs = reshape(SubNbrs, [N, B]);
Cn      = cov(SubNbrs);
fprintf('\nCondition Number of Cn Before Diagonal Load= %f\n', cond(Cn));
if(DIAGLOAD)
    Cn = Cn +eye(size(Cn)).*(DIAGLOAD*max(Cn(:)));
end
fprintf('Condition Number of Cn  After Diagonal Load= %f\n', cond(Cn));

%%
%%% CALCULATE OBSERVATION COVARIANCE %%%
XVecs = reshape(permute(X, [2,1,3]), [N, B]);
Cx     = cov(XVecs);
fprintf('\nCondition Number of Cx Before Diagonal Load= %f\n', cond(Cx));
if(DIAGLOAD>eps)
    Cx = Cn +eye(size(Cx)).*(DIAGLOAD*max(Cx(:)));
end
fprintf('Condition Number of Cx After  Diagonal Load= %f\n', cond(Cx));
%%
%%% CALCULATE LEFT EIGENVECTOR & EIGENVALUES OF Cn,inv*C %%%
if(strcmp(Method, 'ConstructEig'))
    [U, Lambda] = svd(Cx);
    OneOverSqRtLambda  = pinv(sqrt(Lambda));
    Dnt         = OneOverSqRtLambda*U'*Cn*U*OneOverSqRtLambda;
    [V,Dn]      = svd(Dnt);
    W           = V'*OneOverSqRtLambda*U';
elseif(strcmp(Method, 'DirectEig'))
    [V,D,W]  = eig(Cn*pinv(Cx));
    
elseif(strcmp(Method, 'GenEig'))
    [V,D,W] = eig(Cn,Cx);
end

%%
%%% SUBTRACT MEAN %%%
if(ZEROMEAN)
    Xmu    = mean(XVecs);
    XmuBig = repmat(Xmu, [N, 1]);
    XVecs  = XVecs - XmuBig;
end
%%
%%% CALCULATE TRANSFORM %%%
%%% ROWS OF W' ARE LEFT EIGENVECTORS OF R = Cn*pinv(Cx)
%%% COMPUTING W'* R = (R'*W)' SO WE COMPUTE Xvecs*W;
s        = 1;
e        = NComps;
YVecs    = XVecs*W(:, s:e);
Winv     = inv(W);
if(ZEROMEAN)
    transform.Xmu        = Xmu;
end
transform.W          = W;
transform.dim        = NComps;
transform.parameters = Parameters;

%%% RECONSTRUCT ORIGINAL IMAGE AND CALCULATE ERROR STATISTICS %%%
ReconX     = YVecs*Winv(s:e, :);
Err        = ReconX-XVecs;
RMSE       = sqrt(mean(Err(:).*Err(:)));
fprintf('\nRMS Error = %8.4f\n', RMSE);
MinErr = min(Err(:));
MaxErr = max(Err(:));
StpSz  = (MaxErr-MinErr)/20;
Domain = MinErr:StpSz:MaxErr;
%figure(13579);hist(Err(:), Domain);title('Reconstruction Error Histogram')


%%
%%% TURN Y and ReconX BACK INTO AN IMAGES
% Y      = reshape(YVecs, [NRows, NCols,NComps]);
Y      = permute(reshape(YVecs, [NCols, NRows,NComps]), [2,1,3]);
% ReconX = reshape(ReconX, [NRows, NCols,B]);
ReconX = permute(reshape(ReconX, [NCols, NRows,B]), [2,1,3]);
