 % function [Y, transForm] = ReduceDim(X, W, N, downSamRate)
function [Y, WY, transForm] = ReduceDim(X, W, N)
downSapRate   = floor(size(X,2)/N);
if mod(downSapRate,2)==0
    downSapRate = downSapRate + 1;
end
%function for dimensionality reduction
%   Inputs: X: Spectra
%           W: Wavelengths
%           N: Number of reduced dimensions
%  Outputs: Y: 4 Sets of reduced dimensionality spectra, 1 for each algorithm
%           WY: Output Wavelengths
%           U: the transformation for PCA
    %% PCA
    [Y{1}, transForm{1}] = PCAbyDG(X', N); % PCA dimensionality reduction

    %% MNF
%     mnfParameters        = MNFbyDGParameters(N);
%     [Y{2}, transForm{2}] = MNFbyDGu(X, mnfParameters);
%     transForm{2}         = WW(:, 1:mnfParameters.NComps);
    %% Hierarchical Dimensionality Reduction
    hdrParameters                                              = HierarchClusterByDGParameters(N);
    [transForm{3}, ClustsFlag, ClustMems, Y{3}, KLDMat, Hists] = HierarchClusterByDG(X, W, hdrParameters);

    %% Downsampling by building a transformation on the entire training set
    [Y{4}, WY{4}, transForm{4}] = SpectralDwnSmpu(X, W, downSapRate); % for this method, the downsample rate must be odd
end

