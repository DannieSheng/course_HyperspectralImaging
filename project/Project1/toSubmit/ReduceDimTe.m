function Y = ReduceDimTe( X, Xim, transform )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[N, B] = size(X);
%% PCA
Xz   = X'-repmat(transform{1}.mu, [1,size(X',2)]);
yy   = transform{1}.U*Xz;
Y{1} = yy(1:transform{1}.dim,:)';

%% MNF
Sx        = size(Xim);
NRows     = Sx(1);
NCols     = Sx(2);
ZEROMEAN  = transform{2}.parameters.ZEROMEAN;
XVecs     = reshape(permute(Xim, [2,1,3]), [NRows*NCols, B]);
if(ZEROMEAN)
    XmuBig = repmat(transform{2}.parameters.Xmu, [NRows*NCols, 1]);
    XVecs  = XVecs - XmuBig;
end
Y{2}    = XVecs*transform{2}.W(:, 1:transform{2}.parameters.NComps);
Yim{2}  = permute(reshape(Y{2}, [NCols, NRows,transform{2}.parameters.NComps]), [2,1,3]);
% Y{2}      = fliplr((transform{2}.W(:, (end-transform{2}.parameters.NComps+1:end))'*XVecs')');

%% Hierarchical Dimensionality Reduction
Y{3} = zeros(N, transform{3}.parameter.Nc);
% ClustIdx  = find(ClustsFlag);
for c = 1:transform{3}.parameter.Nc
    Y{3}(:, c) = mean(X(:, transform{3}.clusts(c).Idx), 2);
end

%% Downsampling by building a transformation on the entire training set
DsX = conv2(X, transform{4}.ConvMask, 'same');
    
    %%% ADJUST FOR ERRORS AT EDGES OF IMAGE
HalfWindSz                     = floor(transform{4}.DwnSmpRate/2);
DsX(:, 1:HalfWindSz)           = X(:, 1:HalfWindSz);
DsX(:, (end-HalfWindSz+1):end) = X(:, (end-HalfWindSz+1):end);
    
    %%% PICK EVERY nth WAVELENGTH %%%
Y{4} = DsX(:, transform{4}.DwnSmpRate:transform{4}.DwnSmpRate:end);

end

