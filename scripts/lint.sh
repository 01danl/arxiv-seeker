#!/usr/bin/env bash
set -e
flake8 arxiv_seeker --max-line-length=110
mypy arxiv_seeker --ignore-missing-imports
