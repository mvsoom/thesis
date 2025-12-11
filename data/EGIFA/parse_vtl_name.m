function class_cell = parse_vtl_name(name)

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

class_cell = cell(4, 1);

position = 1;
if name(position+(0:7)) == 'pressed_'
  class_cell{1} = '1';
  position = position + 8;
elseif name(position+(0:16)) == 'slightly_pressed_'
  class_cell{1} = '2';
  position = position + 17;
elseif name(position+(0:5)) == 'modal_'
  class_cell{1} = '3';
  position = position + 6;
elseif name(position+(0:16)) == 'slightly_breathy_'
  class_cell{1} = '4';
  position = position + 17;
elseif name(position+(0:7)) == 'breathy_'
  class_cell{1} = '5';
  position = position + 8;
end

if name(position+(0:7)) == 'e_colon_'
  class_cell{2} = 'e_colon';
  position = position + 8;
elseif name(position+(0:1)) == 'i_'
  class_cell{2} = 'i';
  position = position + 2;
elseif name(position+(0:1)) == 'e_'
  class_cell{2} = 'e';
  position = position + 2;
elseif name(position+(0:1)) == 'a_'
  class_cell{2} = 'a';
  position = position + 2;
elseif name(position+(0:1)) == 'o_'
  class_cell{2} = 'o';
  position = position + 2;
elseif name(position+(0:1)) == 'u_'
  class_cell{2} = 'u';
  position = position + 2;
end

if name(position+(0:4)) == '90hz_'
  class_cell{3} = '1';
  position = position + 5;
elseif name(position+(0:5)) == '120hz_'
  class_cell{3} = '2';
  position = position + 6;
elseif name(position+(0:5)) == '150hz_'
  class_cell{3} = '3';
  position = position + 6;
elseif name(position+(0:5)) == '180hz_'
  class_cell{3} = '4';
  position = position + 6;
elseif name(position+(0:5)) == '210hz_'
  class_cell{3} = '5';
  position = position + 6;
end

if name(position+(0:5)) == '500pa.'
  class_cell{4} = '1';
  position = position + 6;
elseif name(position+(0:5)) == '708pa.'
  class_cell{4} = '2';
  position = position + 6;
elseif name(position+(0:6)) == '1000pa.'
  class_cell{4} = '3';
  position = position + 7;
elseif name(position+(0:6)) == '1414pa.'
  class_cell{4} = '4';
  position = position + 7;
elseif name(position+(0:6)) == '2000pa.'
  class_cell{4} = '5';
  position = position + 7;
end
