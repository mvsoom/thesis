%% MATLAB <-> Python venv smoke test

% 

% --- locate venv from env ---
venv = getenv("PROJECT_VENV_PATH");
assert(~isempty(venv), "PROJECT_VENV_PATH not set");

pyexe = fullfile(venv, "bin", "python");

% --- initialize Python deterministically ---
pe = pyenv;
if pe.Status == "NotLoaded"
    pyenv( ...
        "Version", pyexe, ...
        "ExecutionMode", "OutOfProcess" ...
    );
else
    assert(strcmp(pe.Version, pyexe), "Wrong Python already loaded");
end

% --- add project src to Python path ---
proj = getenv("PROJECT_ROOT_PATH");
assert(~isempty(proj), "PROJECT_ROOT_PATH not set");

src = fullfile(proj, "src");

if count(py.sys.path, src) == 0
    insert(py.sys.path, int32(0), src);
end

% --- import module ---
stats = py.importlib.import_module("utils.stats");

% --- call function ---
p = single([0.2 0.3 0.5]);
q = single([0.1 0.4 0.5]);

kl = stats.kl_div(p, q);

fprintf("KL divergence = %.6f\n", double(kl));
