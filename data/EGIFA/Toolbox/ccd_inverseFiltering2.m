function GlottalSource = ccd_inverseFiltering2(sp,fs,gci)

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

F0min=80;
F0max=240;

% Pitch tracking using a method based on the Summation of the Residual Harmonics
[f0,VUVDecisions,SRHVal] = SRH_PitchTracking(sp,fs,F0min,F0max);

% Force everything to be voiced!  Kind of true for this project...
% Have to "guess" what f0 is when n ot found by SRH_PitchTracking
VUVDecisions(1:end)=1;
f0 = max(F0min, f0);

VUVDecisions2=zeros(1,length(sp));
HopSize=round(10/1000*fs);
for k=1:length(VUVDecisions)
    VUVDecisions2((k-1)*HopSize+1:k*HopSize)=VUVDecisions(k);    
end

if nargin < 3
    f0_tmp=f0.*VUVDecisions;
    pos= f0_tmp~=0;
    f0_tmp=f0_tmp(pos);
    F0mean=mean(f0_tmp);
    gci = SEDREAMS_GCIDetection(sp,fs,F0mean);
end;

[GlottalSource] = CCD_GlottalFlowEstimation(sp,fs,gci,f0,VUVDecisions);