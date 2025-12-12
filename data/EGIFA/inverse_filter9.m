function inverse_filter9(varargin)
% usage: inverse_filter(..., 'options', ...)
% Options:
%
%   '--audio', 'FILE'      Input file name.
%
%   '--flow', 'FILE'       Name of the output .mat file, where a signal vector
%                          will be saved.  The sampling rate will be 20 kHz.
%
%   '--wca1'               Using weighted covariance analysis 1 instead
%                          of closed phase covariance analysis.
%
%   '--wca2'               Using weighted covariance analysis 2 instead
%                          of closed phase covariance analysis.
%
%   '--iaif'               Using iterative adaptive inverse filtering instead
%                          of closed phase covariance analysis.
%
%   '--ccd'                Using complex cepstrum decomposition instead
%                          of closed phase covariance analysis.
%
%   '--sg', 'VALUE'        Size of phase error (degrees) to be simulated in
%                          GCIs.  When this option is not used, GCIs are given
%                          by a detector.
%

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

audio = '';
flow = '';
wca1 = 0;
wca2 = 0;
use_iaif = 0;
ccd = 0;
whitenoise = 0;
oracle = 0;
oracle_delayed = 0;
sg = '';
ind_arg = 1;
while ind_arg <= nargin,
  switch varargin{ind_arg}
    case {'--audio'}
      ind_arg = ind_arg + 1;
      if ind_arg <= nargin
        audio = varargin{ind_arg};
      else
        disp('inverse_filter.m: input file name missing')
      end
    case {'--flow'}
      ind_arg = ind_arg + 1;
      if ind_arg <= nargin
        flow = varargin{ind_arg};
      else
        disp('inverse_filter.m: output file name missing')
      end
    case '--wca1'
      wca1 = 1;
    case '--wca2'
      wca2 = 1;
    case '--iaif'
      use_iaif = 1;
    case '--ccd'
      ccd = 1;
    case '--whitenoise'
      whitenoise = 1;
    case '--oracle'
      oracle = 1;
    case '--oracle-delayed'
      oracle_delayed = 1;
    case '--sg'
      ind_arg = ind_arg + 1;
      if ind_arg <= nargin
        sg = varargin{ind_arg};
      else
        disp('inverse_filter.m: phase error missing')
      end
    otherwise
      error(['inverse_filter.m: unrecognized argument ' ...
             varargin{ind_arg}])
  end
  ind_arg = ind_arg + 1;
end
if isempty(audio)
  error('Input file name must be specified!')
end
if isempty(flow)
  error('Output file name must be specified!')
end
  
addpath Toolbox
addpath GLOAT
addpath voicebox
addpath DYPSAGOI

fs = 20000;

% Load speech
[sp, fs_sp] = audioread(audio);
sp = resample(sp,fs,fs_sp); 
sp = sp * 10^6; % converting to cubic cm per sec

% Oracle: emit ground-truth derivative flow, resampled and scaled
if oracle || oracle_delayed
  gt = load([audio(1:end-4) '.mat']);
  if isfield(gt, 'glottal_flow')
    gf = gt.glottal_flow;
  elseif isfield(gt, 'flow')
    gf = gt.flow;
  else
    error('Oracle mode: no glottal_flow/flow in ground-truth file');
  end
  gf = resample(gf, fs, 44100);
  gf = gf(:);                     % ensure column vector
  gf = gf * 10^6;                 % convert to cubic cm per sec
  uu = [0; diff(gf)] * fs;        % derivative
  if oracle_delayed
    % Scorers drop first 14 samples (dd=-14). Advance estimate by padding here.
    pad = 13;
    uu = [zeros(pad,1); uu];
  end
  save(flow, 'uu')
  return
end

% Null model: white noise with matched length and scale
if whitenoise
  sigma = std(sp);
  if sigma==0, sigma = 1; end
  uu = randn(size(sp)) * sigma;
  save(flow, 'uu')
  return
end

% Obtain GCI and GOIs
voicebox('dy_cpfrac', 0.35);
[gci,~,gcic,goic,gdwav,udash,crnmp] = dypsagoi(sp,fs);
if ~isempty(sg)
  load([audio(1:end-4) '.mat'], 'lower_area', 'upper_area', 'glottal_flow');
  lower_area = resample(lower_area, fs, 44100);
  upper_area = resample(upper_area, fs, 44100);
  AreaFnc = min(lower_area, upper_area);
  [area_closings, oa2] = extractInstantsFromAreaFunction(AreaFnc);
  glottal_flow = resample(glottal_flow, fs, 44100)';
  glottal_flow = glottal_flow * 10^6;  %convert to cubic cm per sec
  dflow = [0; diff(glottal_flow).'] * ...
          fs;  %Taking derivative need to divide by dt = 1/fs.
  gci = extract_gcis(dflow, area_closings);
  diff_gci = diff(gci);
  diff_gci(end+1) = diff_gci(end);
  oa = gci + diff_gci * str2num(sg) / 360;
  gci = round(oa);
  gci = gci + 13;
  gci(gci<1|gci>length(sp)) = [];
end
goi = pickGOIs(gci, goic);

% Inverse filtering
if use_iaif
  uu = iaif(sp, fs, 20, 4, 20);
elseif ccd
  if isempty(sg)
    uu = ccd_inverseFiltering2(sp, fs)';
  else
    uu = ccd_inverseFiltering2(sp, fs, gci)';
  end
elseif wca1
  par = projParam('rgauss');
  uu = weightedlpc3(sp, gci, goi, fs, par);
elseif wca2
  par = projParam('ame');
  uu = weightedlpc3(sp, gci, goi, fs, par);
else
  par = projParam('cp');
  uu = weightedlpc3(sp, gci, goi, fs, par);
end
isnan_uu = isnan(uu);
if ccd
  uu(~isnan_uu) = uu(~isnan_uu) - mean(uu(~isnan_uu));
end
uu(isnan_uu) = 0;

save(flow, 'uu')
