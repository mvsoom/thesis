function par = projParam(wparmethod)

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

if nargin == 0
    wparmethod = 'ame';
end;

% Parameters for weights
par.wpar.method = wparmethod;
par.wpar.fs = 20000;

switch par.wpar.method
    case 'cp'
        par.wpar.minF0 = 50;  % Hz (assume that no voice has lower f0)
        par.wpar.cpFrac = 0.8;  % Assumed proportion of closed phase in a cycle
        par.wpar.cpDelay = 0.9e-3; % in s. Closed analysis begins after GCI
    case 'ame'
        par.wpar.minF0 = 50;  % Hz (assume that no voice has lower f0)
        par.wpar.d = 0.05;  % Amplitude parameter 0.01  (SEE: Fig 1 in Alku et.al "Improved formant frequency ...")
        par.wpar.DQ = 0.4;  % Duration quotient  0.4
        par.wpar.PQ = 0.8;  % Position quotient 0.8
        par.wpar.rlen =  9;   % Ramp length   3
    case 'rgauss'
        par.wpar.kappa = 0.9;
        par.wpar.sig=sqrt(50);
    otherwise 
        error('Unknown weighted LP method');
end


%par.wpar.minF0 = 50;  % Hz (assume that no voice has lower f0)
%par.wpar.cpFrac = 0.2;  % Assumed proportion of closed phase in a cycle
%par.wpar.cpDelay = 0.5e-3; % in s. Closed analysis begins after GCI


% Parameters for determining frame boundaries  VALUES FROM IAIF :
par.fpar.wl = 32e-3;  % window size in samples (25 ms)
par.fpar.inc = 16e-3; % frame increment in samples (10 ms)

% LPC modeling parameters
par.mpar.f_preemph = 10; % Hz  (Preemphasis cutoff)
% par.mpar.windowFunction = @hamming;
par.mpar.fade = 0;
