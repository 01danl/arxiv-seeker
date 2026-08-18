# Contributing to ArxivSeeker

Thanks for considering a contribution! This project aims for clean, well-tested,
modular code.

## Setup

```bash
git clone <fork-url>
cd arxiv-seeker
python -m venv venv && source venv/bin/activate
pip install -e ".[dev,rag,web]"
pre-commit install  # optional but recommended
```

## Workflow

1. Fork and branch from `main`.
2. Write tests for new behavior (`tests/`) — aim to keep network/GPU-free tests mockable.
3. Run `scripts/format.sh` and `scripts/lint.sh` before opening a PR.
4. Run `pytest --cov=arxiv_seeker` and keep coverage from regressing.
5. Open a PR with a clear description of the change and why.

## Code style

- `black` (line length 110) + `isort` for formatting.
- `flake8` for linting, `mypy` for type checking on new modules.
- Prefer small, focused functions; dataclasses for structured data (see `api_client.Paper`,
  `rag/chunker.Chunk`).

## Areas that could use help

See the Roadmap in `README.md` — citation graph visualization, cross-encoder reranking,
FastAPI layer, and Zotero export are good starting points.
