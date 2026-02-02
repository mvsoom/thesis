% dump_gci_indices.m
% Compute GCI indices via extract_gcis and append to EGIFA .mat files.

root = fileparts(mfilename('fullpath'));
subdirs = {'speech', 'vowel'};

% Match inverse_filter9.m (sg branch) defaults
fs = 20000;
src_fs = 44100;
sg = 0;          % phase error (degrees)
advance = 13;    % sample advance applied after sg shift

% Output variable handling
out_var = 'gci';
alt_var = 'gci_extract';
overwrite = false;
use_audio_length = true;

addpath(root);
addpath(fullfile(root, 'Toolbox'));

for d = 1:numel(subdirs)
  mat_files = dir(fullfile(root, subdirs{d}, '*.mat'));
  for i = 1:numel(mat_files)
    mat_path = fullfile(mat_files(i).folder, mat_files(i).name);

    vars = whos('-file', mat_path);
    var_names = {vars.name};

    out_name = out_var;
    if any(strcmp(var_names, out_var))
      if overwrite
        out_name = out_var;
      elseif ~isempty(alt_var) && ~any(strcmp(var_names, alt_var))
        out_name = alt_var;
      else
        fprintf('skip (vars exist): %s\n', mat_path);
        continue;
      end
    end

    s = load(mat_path);
    if ~isfield(s, 'lower_area') || ~isfield(s, 'upper_area') || ~isfield(s, 'glottal_flow')
      fprintf('skip (missing vars): %s\n', mat_path);
      continue;
    end

    lower_area = resample(s.lower_area, fs, src_fs);
    upper_area = resample(s.upper_area, fs, src_fs);
    AreaFnc = min(lower_area, upper_area);
    [area_closings, ~] = extractInstantsFromAreaFunction(AreaFnc);
    if numel(area_closings) < 2
      fprintf('skip (few area_closings): %s\n', mat_path);
      continue;
    end

    glottal_flow = resample(s.glottal_flow, fs, src_fs)';
    glottal_flow = glottal_flow * 10^6;
    dflow = [0; diff(glottal_flow).'] * fs;

    gci = extract_gcis(dflow, area_closings);

    if numel(gci) >= 2
      diff_gci = diff(gci);
      diff_gci(end+1) = diff_gci(end);
      gci = round(gci + diff_gci * (sg / 360));
    end
    gci = gci + advance;

    n = length(dflow);
    if use_audio_length
      wav_path = [mat_path(1:end-4) '.wav'];
      if exist(wav_path, 'file') == 2
        [sp, fs_sp] = audioread(wav_path);
        sp = resample(sp, fs, fs_sp);
        n = length(sp);
      end
    end
    gci(gci < 1 | gci > n) = [];

    out_struct = struct();
    out_struct.(out_name) = gci;
    save(mat_path, '-struct', 'out_struct', '-append');

    fprintf('saved %s -> %s\n', out_name, mat_path);
  end
end
