# BNGIF

## Installation

Requires `python>=3.12`.

Clone repo, enable environment, install:
```bash
git clone https://github.com/mvsoom/BNGIF.git
cd BNGIF
python -m venv .venv && source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -U jax # or "jax[cuda13]" if GPU
pip install -e .
```

Then, in the current root folder, activate environment variables:
```bash
sudo apt install direnv
direnv allow
```
There are convenient plugins for `vscode` to have the `.envrc` file loaded automatically, which will expose the environment variables in the Jupyter notebooks.

For evaluation experiments, install latest `matlab` version.
Then, with venv activated, do:
```bash
pip install matlabengine
```