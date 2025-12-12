function test_vtl3(varargin)
% usage: test_vtl(..., 'options', ...)
% Options:
%
%   '--data', 'FOLDER'           Directory containing the VTL data.
%
%   '--res', 'FOLDER'            Directory for storing the results.  This folder
%                                should be created as a sub-folder of the data
%                                folder.
%
%   '--a_only'                   Testing on utterances of /a/ only.
%
%   any other option             Passed down to inverse_filter.m.
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

prog = 'inverse_filter9';

passed_down = {};
data = '';
res = '';
a_only = 0;
ind_arg = 1;
while ind_arg <= nargin,
  switch varargin{ind_arg}
    case {'--data'}
      ind_arg = ind_arg + 1;
      if ind_arg <= nargin
        data = varargin{ind_arg};
      else
        disp('test_vtl.m: data folder missing')
      end
    case {'--res'}
      ind_arg = ind_arg + 1;
      if ind_arg <= nargin
        res = varargin{ind_arg};
      else
        disp('test_vtl.m: result folder missing')
      end
    case '--a_only'
      a_only = 1;
    otherwise
      passed_down = [passed_down {varargin{ind_arg}}];
  end
  ind_arg = ind_arg + 1;
end
if isempty(data)
  error('Data folder must be specified!')
end
if isempty(res)
  error('Result folder must be specified!')
end
if ~exist(res, 'dir')
  mkdir(res);
end

% Collect files to process (for progress reporting)
files = {};
fid = fopen([data 'ls.out'], 'rt');
file = fgetl(fid);
while ischar(file),
  in_subset = 1;
  class_cell = parse_vtl_name(file);
  if a_only & ~strcmp(class_cell{2}, 'a')
    in_subset = 0;
  end
  if in_subset
    files{end+1} = file; %#ok<AGROW>
  end
  file = fgetl(fid);
end
fclose(fid);

nfiles = numel(files);
fprintf('Processing %d files...\n', nfiles);

for ind = 1:nfiles
  file = files{ind};
  audio = [data file(1:end-4) '.wav'];
  flow = [res file(1:end-4) '.mat'];
  list = [{prog} {'--audio'} {audio} {'--flow'} {flow} passed_down];
  feval(list{:});
  if mod(ind, max(1, floor(nfiles/100))) == 0 || ind == nfiles
    fprintf('  %d/%d (%.1f%%)\r', ind, nfiles, 100*ind/nfiles);
  end
end
if nfiles > 0
  fprintf('\n');
end
