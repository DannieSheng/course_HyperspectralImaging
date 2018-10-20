 function [SpecDist] = PlotSpectraDistributionVer2(Spectra, Wvs,SampInt, FigNum);
%function [SpecDist] = PlotSpectraDistributionVer2(Spectra, Wvs,SampStuff, FigNum);
%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% A FUNCTION THAT CREATES & DISPLAYS SPECTRA AS A 2D HISTOGRAM   %%%
%%%    SPECTRA ARE ASSUMED REFLECTANCES OR EMISSIVITIES IN [0,1]   %%%
%%%    SPECTRA ARE MAPPED TO INTEGERS BETWEEN 0 AND 100 (OR < 100) %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                                                         %%%
%%% INPUTS:                                                 %%%
%%%   I1. Spectra: NUMBER SPECTRA x NUMBER BANDS...         %%%
%%%       ...ARRAY OF REFLECTANCES OR EMISSIVITIES          %%%
%%%   I2. Wvs:     A VECTOR OF THE SPECTRAL WAVELENGTHS     %%%
%%%   I3. SampInt: FRACTIONAL SIZE OF HISTOGRAM BINS        %%%
%%%   I4. FigNum:  INDEX OF FIGURE FOR DISPLAY              %%%
%%%          IF FigNum < 1, DO NOT DISPLAY ANYTHING         %%%
%%%                                                         %%%
%%% OUTPUTS:                                                %%%
%%%   O1. SpecDist IS THE 2D HISTOGRAM                      %%%
%%%                                                         %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                                                         %%%
%%% AUTHOR:      Darth Gader                                %%%
%%% LAST UPDATE: 090218                                     %%%
%%%                                                         %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
%%% INITIALIZE PARAMETERS %%%
TopVal        = min(1, max(Spectra(:))+0.05);
IntTopReflect = round(TopVal*100);
IntSampInt    = min(100, round(SampInt*100));
SMOOTHSIZE    = [3,3];
NumSpec       = size(Spectra, 1);
NWvs          = size(Spectra, 2);
Domain        = 1:IntSampInt:IntTopReflect;
NumBins       = length(Domain);
SpecDist      = zeros(NumBins, NWvs);
assert(NWvs == length(Wvs), 'Wavelength sizes don''t match');

%%% MAP SPECTRA TO [0, 100] %%%
MappedSpectra = min(100, (Spectra*99)+1);
MappedSpectra = max(1,   round(MappedSpectra/SampInt)*SampInt);

%%
%%% MAKE A HISTOGRAM FOR EACH WAVELENGTH %%%
for Lambda = 1:NWvs;
    SpecDist(:, Lambda) = hist(MappedSpectra(:, Lambda), Domain);
end

%%% SMOOTH BY TAKING A LOCAL MAX FOLLOWED BY A LOCAL AVERAGE %%%
SpecDist   = ordfilt2(SpecDist, 9, ones(3,3));
SpecDist   = conv2(SpecDist, (1/prod(SMOOTHSIZE))*ones(SMOOTHSIZE), 'same');

%%
%%% DISPLAY AS MESH %%%
if(FigNum > 0)
    SpecDist = vertcat(SpecDist, zeros(1,NWvs));
    Domain   = horzcat([0], Domain)
    XAxis    = Wvs;
    YAxis    = Domain';
    figure(FigNum);
    mesh(XAxis, YAxis, SpecDist);
    xlim([Wvs(1), Wvs(end)])
    xlabel('Wavelength')
    ylabel('Reflectance')
    view([-194, 34])
end
%%% END OF FUNCTION %%%
%%%%%%%%%%%%%%%%%%%%%%%