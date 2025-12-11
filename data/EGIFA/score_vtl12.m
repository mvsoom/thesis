function score_vtl12(varargin)
% Usage: score_vtl('folder', ..., 'options', ...)
%
% Options:
%
%   'FOLDER'             Directory containing derivative flow estimates.
%
%   '--v2.0'             For VTL2.0.
%
%   '--br', 'NUM'        Specifying a flow shape subset with breathiness level
%                        1-5.
%
%   '--pr', 'NUM'        Specifying a subset with subglottal pressure level 1-5.
%
%   '--f0', 'NUM'        Specifying an F0 subset with frequency level 1-5.
%
%   '--vw', 'VW'         Specifying a vowel subset ('i', 'e', 'e_colon', 'a',
%                        'o', or 'u').
%
%   '--cfl', 'FREQ'      Specifying a cutoff frequency (Hz) for low-pass
%                        filtering.
%
%   '--cfh', 'FREQ'      Specifying a cutoff frequency (Hz) for high-pass
%                        filtering.
%
%   '--h1h2'             Using the H1-H2 feature.
%
%   '--hrf'              Using the HRF feature.
%
%   '--naq'              Using the NAQ feature.
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

folder = '';
v20 = 0;
br = '';
pr = '';
f0 = '';
vw = '';
cfl = '';
cfh = '';
h1h2 = 0;
hrf = 0;
naq = 0;
ind_arg = 1;
while ind_arg <= nargin,
  switch varargin{ind_arg}
    case '--v2.0'
      v20 = 1;
    case '--br'
      ind_arg = ind_arg + 1;
      if ind_arg <= nargin
        br = varargin{ind_arg};
      else
        error('score_vtl.m: breathiness level missing')
      end
    case '--pr'
      ind_arg = ind_arg + 1;
      if ind_arg <= nargin
        pr = varargin{ind_arg};
      else
        error('score_vtl.m: pressure level missing')
      end
    case '--f0'
      ind_arg = ind_arg + 1;
      if ind_arg <= nargin
        f0 = varargin{ind_arg};
      else
        error('score_vtl.m: frequency level missing')
      end
    case '--vw'
      ind_arg = ind_arg + 1;
      if ind_arg <= nargin
        vw = varargin{ind_arg};
      else
        error('score_vtl.m: vowel missing')
      end
    case '--cfl'
      ind_arg = ind_arg + 1;
      if ind_arg <= nargin
        cfl = varargin{ind_arg};
      else
        error('score_vtl.m: cutoff frequency missing')
      end
    case '--cfh'
      ind_arg = ind_arg + 1;
      if ind_arg <= nargin
        cfh = varargin{ind_arg};
      else
        error('score_vtl.m: cutoff frequency missing')
      end
    case '--h1h2'
      h1h2 = 1;
    case '--hrf'
      hrf = 1;
    case '--naq'
      naq = 1;
    otherwise
      folder = varargin{ind_arg};
  end
  ind_arg = ind_arg + 1;
end
if isempty(folder)
  error('Result folder must be specified!')
end

fs = 20000;
addpath Toolbox

fid = fopen([folder '../ls.out'], 'rt');
file = fgetl(fid);
ind = 0;
while ischar(file),
  in_subset = 1;
  class_cell = parse_vtl_name(file);
  if ~isempty(br)
    if ~strcmp(class_cell{1}, br)
      in_subset = 0;
    end
  elseif ~isempty(pr)
    if ~strcmp(class_cell{4}, pr)
      in_subset = 0;
    end
  elseif ~isempty(f0)
    if ~strcmp(class_cell{3}, f0)
      in_subset = 0;
    end
  elseif ~isempty(vw)
    if ~strcmp(class_cell{2}, vw)
      in_subset = 0;
    end
  end
  if in_subset
    if v20
      load([folder '../' file(1:end-4) '.mat'], 'flow');
      glottal_flow = flow;
    else
      load([folder '../' file(1:end-4) '.mat'], 'glottal_flow');
    end
    glottal_flow = resample(glottal_flow, fs, 44100)';
    glottal_flow = glottal_flow * 10^6;  %convert to cubic cm per sec
    dflow = [0; diff(glottal_flow).'] * ...
            fs;  %Taking derivative need to divide by dt = 1/fs.
    
    load( [folder file(1:end-4) '.mat'], 'uu' );
    dd = -14;
    signal1 = dflow;
    signal2 = uu(-dd:end);
    oa = min( length(signal1), length(signal2) );
    interval = round(oa*0.1) : round(oa*0.7);
    signal1 = signal1(interval);
    signal2 = signal2(interval);
    signal1 = signal1(:); signal2 = signal2(:);
    rms1 = sqrt(mean(signal1.^2));
    if ~isempty(cfl)
      [b, a] = low_pass(str2num(cfl));
      signal1 = filter(b, a, signal1);
      signal2 = filter(b, a, signal2);
    elseif ~isempty(cfh)
      [b, a] = high_pass(str2num(cfh));
      signal1 = filter(b, a, signal1);
      signal2 = filter(b, a, signal2);
    end

    % cycle-level error evaluation
    load([folder '../' file(1:end-4) '.mat'], 'lower_area', 'upper_area');
    lower_area = resample(lower_area, fs, 44100);
    upper_area = resample(upper_area, fs, 44100);
    AreaFnc = min(lower_area, upper_area);
    [oa1, oa2] = extractInstantsFromAreaFunction(AreaFnc);
    gci = oa1 - interval(1) + 1;
    gci = gci( gci>=1 & gci<=length(interval) );
    if h1h2 | hrf | naq
      [oa1, oa2, oa3, oa4, oa5] = ...
          extractVoiceFeatures2(cumsum(signal1/fs), fs, gci);
      if h1h2
        feature1 = oa4;
      elseif hrf
        feature1 = oa5;
      else % naq
        feature1 = oa3;
      end
      [oa1, oa2, oa3, oa4, oa5] = ...
          extractVoiceFeatures2(cumsum(signal2/fs), fs, gci);
      if h1h2
        feature2 = oa4;
      elseif hrf
        feature2 = oa5;
      else % naq
        feature2 = oa3;
      end
      absolute_error_cycles = abs(feature1 - feature2);
      signed_error_cycles = feature2 - feature1;
    else
      number_cycles = length(gci) - 1;
      absolute_error_cycles = zeros(number_cycles, 1);
      signed_error_cycles = zeros(number_cycles, 1);
      for i1 = 1:number_cycles,
        T = gci(i1+1) - gci(i1);
        nn = (0:T-1) + gci(i1);
        signal1_cycle = signal1(nn);
        signal2_cycle = signal2(nn);
        gain = (signal1_cycle.'*signal2_cycle) / (signal2_cycle.'*signal2_cycle);
        if gain < 0
          gain = 0;
        end
        signal2_cycle = signal2_cycle * gain;
        absolute_error_cycles(i1) = median(abs(signal1_cycle-signal2_cycle)) ...
            / rms1;
        signed_error_cycles(i1) = median(signal2_cycle-signal1_cycle) / rms1;
      end
    end

    absolute_error(ind+1) = median(absolute_error_cycles);
    signed_error(ind+1) = median(signed_error_cycles);
    ind = ind + 1;
  end
  file = fgetl(fid);
end
fclose(fid);

disp([ 'Number of samples: ' num2str(length(absolute_error)) ])
error_string = num2str(mean(absolute_error));
if h1h2 | hrf
  disp([ 'Average median absolute error: ' error_string ' (dB)' ])
elseif naq
  disp([ 'Average median absolute error: ' error_string ])
else
  disp([ 'Average normalized median absolute error: ' error_string ])
end
disp([ '(standard deviation: ' num2str(std(absolute_error)) ')' ])
error_string = num2str(mean(signed_error));
if h1h2 | hrf
  disp([ 'Average median error: ' error_string ' (dB)' ])
elseif naq
  disp([ 'Average median error: ' error_string ])
else
  disp([ 'Average normalized median error: ' error_string ])
end
disp([ '(standard deviation: ' num2str(std(signed_error)) ')' ])
