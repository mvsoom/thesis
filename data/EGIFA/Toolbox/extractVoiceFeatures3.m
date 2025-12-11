function [mfdr, pa, naq, h1h2, hrf] = extractVoiceFeatures3(u, fs, gci)

%EXTRACTVOICEFEATURES extracts voice features from the glottal flow
%derivative  [mfdr, cq, pa, naq] = extractVoiceFeatures(u, fs, gci)
%
%  Inputs:  u        is the glottal flow in [cm^3/s]
%           fs       is the sampling frequency in Hz
%           gci      is the glottal closure instants indices (samples) of
%                    length Ng
%
% Outputs: 
% Features defined in Patel 2011 that are derived from glottal flow
% of length Ng
%
% MFDR  - Maximum Flow Declination Rate  [l/s^2]
% CQ    - Closed Quotient  [unitless]
% PA    - Pulse Amplitude  [cm^3/s]
% NAQ   - Normalized Amplitude Quotient [unitless]
%
% ___________________________
% not sure about these
% H1-H2  - Level difference between first and second harmonics
% Alpha  - Ratio between the summed energy between 50Hz-1kHz and 1-5 kHz
%          (computed from the long term average spectrum (LTAS))
%
% the rest of the parameters are obtained directly from sp
%
% leq   - Equivalent sound level
% Shimmer
% HNR  (harmonic-noise-ratio)
% Jitter
% MF0 (mean f0)

% Copyright 2017 Yu-Ren Chien and Jon Gudnason
%
% This file is part of EGIFA.
%
% EGIFA is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% EGIFA is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with EGIFA.  If not, see <http://www.gnu.org/licenses/>.

uu=[diff(u); 0]*fs;

number_cycles = length(gci) - 1;
mfdr = zeros(number_cycles, 1);
pa = zeros(number_cycles, 1);
naq = zeros(number_cycles, 1);
h1h2 = zeros(number_cycles, 1);
hrf = zeros(number_cycles, 1);

for ig = 1:number_cycles,
  T = gci(ig+1) - gci(ig);
  nn = (0:T-1) + gci(ig);  
  
  uuseg=uu(nn);
  useg=u(nn);
  
  % Maximum flow declination rate
  dpeak = -min(uuseg);  % flow derivative in cm^3/s^2

  % Maximum flow
  fac = max(useg) - min(useg);      % flow in cm^3/s
  
  % Pitch period 
  Ttime=T/fs;           % in seconds
  
  mfdr(ig) = dpeak/1000;   % in liters per second squared
  naq(ig) = fac/(dpeak*Ttime);  % unit check: cm^3/s / (cm^3/s^2 * s) = unitless
  pa(ig) = fac;
  f0 = 1/Ttime;
  
  % Spectral parameters
  oa = abs(fft(useg)) / T;
  number_partials = floor(3000/f0);
  if number_partials > 1
    partial_amplitudes = oa(2:number_partials+1);
    amplitudes_db = 20*log10( partial_amplitudes / partial_amplitudes(1) );
    h1h2(ig) = -amplitudes_db(2);
    hrf(ig) = 10*log10(sum(10.^( amplitudes_db(2:end)/10 )));
  else
    h1h2(ig) = 0;
    hrf(ig) = 0;
  end
end
naq(isnan(naq)) = 0;
h1h2(isnan(h1h2)) = 0;
hrf(isnan(hrf)) = 0;
