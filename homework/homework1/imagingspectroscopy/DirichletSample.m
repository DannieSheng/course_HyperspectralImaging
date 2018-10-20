function [samples] = DirichletSample(alpha, numSamples)
%DIRICHLETSAMPLE returns a number of samples from a given Dirichlet
%distribution.
%
% Syntax: [samples] = DirichletSample(alpha, numSamples)
%
% Inputs:
%   alpha - 1xN array of alpha values describing Dirichlet distribution
%   numSamples - Number of samples to take from Dirichlet distribution
%
% Outputs:
%   samples - numSamplesxN matrix describing the samples taken
%
    if(size(alpha,1) == 1)
        Y = randg(repmat(alpha, [numSamples, 1]));
        samples = Y./repmat(sum(Y')', [1, length(alpha)]);
    else
        Y = randg(alpha);
        samples = Y./repmat(sum(Y')', [1, size(alpha,2)]);
    end
end