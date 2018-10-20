 function [Y, U, L, Mu] = PCAbyDG(X, NComps);
%function [Y, U, L, Mu] = PCAbyDG(X, NComps);
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FUNCTION TO COMPUTE AN MNF TRANSFORM OF THE HYPERSPECTRAL IMAGE X
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% INPUTS:
%%% X IS EITHER:
%%%        (1) a N x B Matrix
%%%           and the columns of X are spectra 
%%%     or
%%%        (2) an NRows x NCols X B Spectral Data Cube
%%%     where
%%%        B = Number of Bands
%%%        N = Number of Spectra
%%%
%%%  NComps IS THE NUMBER OF COMPONENTS TO KEEP.  MUST BE >=1.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% OUTPUTS:
%%%  Y      = 1ST NComps Principal Components sorted by decreasing variance
%%%  U      = B x B matrix of eigenvectors of the covariance matrix
%%%  Lambda = B x 1 vector of eigenvalues  of the covariance matrix
%%%  Mu     = mean of columns of X
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% USAGE:
%%%         To compute transform of N x 1 vector x, use U'*(x-Mu)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Author:  Darth Gader %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
%%% REFORMAT X AS A BxN MATRIX IF IT IS NRows x NCols x B %%%
%%% THROW AN ERROR IF X IS 1D OR >3D %%%
Sx    = size(X);
NDims = min(4,length(Sx)); 
switch NDims
    case 1
        error('Error: Not enough Dimensions');
    case 3
        N = Sx(1)*Sx(2);
        B = Sx(3);
        X = shiftdim(X, 2);
        X = reshape(X, [B, N]);
    case 4
        error('Error: Too many Dimensions');
end

%%% CALCULATE MEAN AND SUBTRACT IT %%%
[N,B] = size(X);
Mu    = mean(X);
BigMu = repmat(Mu, [N, 1]);
Xz    = X-BigMu;

%%
%%% CALCULATE COVARIANCE MATRIX %%%
%C = (1/(N-1))*Xz*Xz';
C = cov(X); %(1/(N-1))*Xz*Xz';
%%
%%% GET EIGENVALUES AND EIGENVECTORS OF COVARIANCE MATRIX %%%
[U, Lambda] = svd(C);
L           = diag(Lambda);

%%% COMPUTE TRANSFORMATION AND REDUCE DIMENSIONALITY
%Y = U*Xz;
Y = Xz*U;
Y = Y(:, 1:NComps);

%%% MAKE MATRIX SIZE CONSISTENT WITH MNFbyDG %%%
%Y = Y';

%%% THE END %%%
end