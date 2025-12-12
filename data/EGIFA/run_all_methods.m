function run_all_methods()
% Batch runner to generate and score all GIF variants (default metric).
% Assumes current working directory is the EGIFA folder.

base = pwd;
addpath(genpath(base));

logfile = fullfile(base, 'scores_summary.txt');
fid = fopen(logfile, 'w');
fprintf(fid, 'EGIFA batch run %s\n', datestr(now));

methods = { ...
  struct('name','cp',        'flags', {{}}), ...
  struct('name','wca1',      'flags', {{'--wca1'}}), ...
  struct('name','wca2',      'flags', {{'--wca2'}}), ...
  struct('name','iaif',      'flags', {{'--iaif'}}), ...
  struct('name','ccd',       'flags', {{'--ccd'}}), ...
  struct('name','whitenoise','flags', {{'--whitenoise'}}) ...
};

datasets = { ...
  struct('name','vowel',  'data','vowel',  'scorer',@score_vtl12), ...
  struct('name','speech', 'data','speech', 'scorer',@score_vtl_continuous1) ...
};

for m = methods
  for d = datasets
    data_folder = [fullfile(base, d{1}.data) filesep];
    res_folder = [fullfile(base, d{1}.data, ['res_' m{1}.name]) filesep];

    fprintf('Running %s on %s...\n', m{1}.name, d{1}.name);
    test_args = [{'--data', data_folder, '--res', res_folder}, m{1}.flags{:}];
    test_vtl3(test_args{:});

    score_cmd = sprintf('%s(''%s'')', func2str(d{1}.scorer), res_folder);
    score_out = evalc(score_cmd);

    fprintf(fid, '\n=== Method: %s | Dataset: %s ===\n', m{1}.name, d{1}.name);
    fprintf(fid, '%s\n', score_out);
  end
end

fclose(fid);
disp(['Scores written to ', logfile]);
