function gci = extract_gcis(uu, area_closings)

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

number_cycles = length(area_closings) - 1;
gci = zeros(1, number_cycles);

for ig = 1:number_cycles,
  T = area_closings(ig+1) - area_closings(ig);
  nn = (0:T-1) + area_closings(ig);  
  uuseg=uu(nn);
  
  [oa1, oa2] = min(uuseg);
  gci(ig) = area_closings(ig) - 1 + oa2;
end
