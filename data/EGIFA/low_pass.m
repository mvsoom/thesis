function [b, a] = low_pass(cutoff)

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

[b, a] = butter(10, cutoff/10e3);

if ~nargout
  normalized_frequencies = 0:0.0025:0.5;
  number_frequencies = length(normalized_frequencies);
  frequency_response = zeros(number_frequencies, 1);
  for i1 = 1:number_frequencies,
    z = exp(i*2*pi*normalized_frequencies(i1));
    c = a / a(1); d = b / a(1);
    numerator = sum( d .* z.^(0:-1:-length(b)+1) );
    denominator = 1 + sum( c(2:end) .* z.^(-1:-1:-length(a)+1) );
    frequency_response(i1) = abs(numerator/denominator);
  end
  plot(normalized_frequencies*20e3, 20*log10(frequency_response))
  xlabel('Frequency (Hz)')
  ylabel('Magnitude (dB)')
end
