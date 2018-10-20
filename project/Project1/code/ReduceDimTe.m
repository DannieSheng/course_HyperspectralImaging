function Y = ReduceDimTe( X, transform )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[N, B] = size(X);
%% PCA
Xz   = X'-repmat(transform{1}.mu, [1,size(X',2)]);
yy   = transform{1}.U*Xz;
Y{1} = yy(1:transform{1}.dim,:)';

%% MNF
% ZEROMEAN  = transform{2}.Parameters.ZEROMEAN;
% if(ZEROMEAN)
%     Xmu    = mean(X);
%     X      = X - repmat(transform.mu, [size(X,1), 1]);
% end
% Y{2}    = X*transform{2}.W(:, 1:transform{2}.parameters.NComps);

%% Hierarchical Dimensionality Reduction
Y{3} = zeros(N, transform{3}.parameter.Nc);
% ClustIdx  = find(ClustsFlag);
for c = 1:transform{3}.parameter.Nc
    Y{3}(:, c) = mean(X(:, transform{3}.clusts(1).Idx), 2);
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

