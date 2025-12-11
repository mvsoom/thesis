function [uu,ar,ee,dc,w]=weightedlpc3(sp, gci, goi, fs, par)

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

nsp = length(sp);
nar = ceil(fs/1000);  % number of LPC poles: 1 pole per 1000 Hz.  Note: don't add the two poles that are normally thought of as representing lip and source


% Determine weight vector
wpar = par.wpar;
wpar.fs =fs;
w = weightsForLP3(gci, goi, nsp, wpar);

% Determine frame boundaries
fpar = par.fpar;
wl = round(fs*fpar.wl);
inc = round(fs*fpar.inc);


tstart = (nar+1):inc:(nsp-wl-1);
tend = tstart+wl;
T=[tstart(:) tend(:)];
T(:,tstart>nsp) = [];
T(:,tend>nsp) = [];

%%% Closed phase covariance analysis %%%
f_preemph = par.mpar.f_preemph; % Hz  (Preemphasis cutoff)
fade= par.mpar.fade;

b=[1 -exp(-2*pi*f_preemph/fs)];
sp_preemph = filter(b,1,sp);      %Estimate the AR on pre-emphasised signal

[ar,ee,dc]=lpccovar(sp_preemph, nar, T, w);
%ar2=ar.*repmat(sqrt(ee(:,1)),1,size(ar,2));
%dc=dc.*sqrt(ee(:,2));
uu=lpcifilt(sp, ar, T, dc,fade);

%u = filter(1,b,uu);
%u = adjustU(u,gci,goi);
