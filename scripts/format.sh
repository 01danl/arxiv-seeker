#!/usr/bin/env bash
set -e
black arxiv_seeker tests examples
isort arxiv_seeker tests examples
