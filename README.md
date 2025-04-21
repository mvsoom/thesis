# BNGGIF

Clone the repo, create a virtual‑env with Python 3.12 (or later), activate it, then install the package in editable mode so all dependencies declared in `pyproject.toml` are fetched automatically:

1. Install repo
```bash
git clone https://github.com/your‑user/bngif.git
cd bngif
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

2. Load .env file (vscode can do this automatically through `python.envFile` setting)