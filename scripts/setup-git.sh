#!/usr/bin/env bash
# setup-git.sh: run once after you clone

git config filter.ipynbclean.clean   "scripts/ipynb-clean.sh"
git config filter.ipynbclean.smudge  "cat"
git config filter.ipynbclean.required true

git config filter.vsnbclean.clean   "scripts/vsnb-clean.sh"
git config filter.vsnbclean.smudge  "cat"
git config filter.vsnbclean.required true