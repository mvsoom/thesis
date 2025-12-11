function [gci, goi] = extractInstantsFromAreaFunction(AreaFnc)

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

%AreaFnc is in cm^2
%thR=0.01;

%AreaFnc is in m^2
thR=1e-6;

lA=length(AreaFnc);
sA=round(0.2*lA);
eA=round(0.8*lA);

AreaFnc=AreaFnc-min(AreaFnc(sA:eA));

inst=diff(AreaFnc>thR);

gci=find(inst==-1);
goi=find(inst==+1);

%Assumtion that gci and goi occur in tandem, just force gci to be first and
%goi to be last in sequence:

goi(goi < gci(1))=[];
gci(gci > goi(end))=[];

% Remove ones close to beginning
ix=median(diff(gci))<gci;
iy=(length(AreaFnc)-median(diff(gci)))>goi;
gci=gci(and(ix,iy))';
goi=goi(and(ix,iy))';