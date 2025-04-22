# BNGIF

## Installation

Clone repo, enable environment, install:
```bash
git clone https://github.com/mvsoom/BNGIF.git
cd BNGIF
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Then, in the current root folder, activate environment variables:
```bash
sudo apt install direnv
direnv allow
```
There are convenient plugins for `vscode` to have the `.envrc` file loaded automatically, which will expose the environment variables in the Jupyter notebooks.