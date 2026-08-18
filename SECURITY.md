# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please **do not** open a public issue.
Instead, email the maintainers (or use GitHub's private vulnerability reporting)
with:

- A description of the vulnerability and its impact
- Steps to reproduce
- Any suggested fix, if you have one

We aim to acknowledge reports within 5 business days.

## Scope notes

- API keys (OpenAI/Anthropic/Semantic Scholar) are read from `.env` only — never commit
  `.env` or hard-code credentials. `.env` is git-ignored by default.
- Downloaded PDFs and FAISS indexes are stored locally under `data/`; treat that directory
  as containing potentially sensitive research data if you index non-public papers via a
  custom PDF path.
