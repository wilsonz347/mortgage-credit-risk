# mortgage-credit-risk

Survival models for mortgage default and prepayment risk (Freddie Mac Single-Family Loan-Level Dataset). Design choices and their reasons are in `docs/decisions.md`. Read it before suggesting design changes.

Status: v0.1.0 done (data, baselines, rate spread, expected loss, README, model card).

## Commands
- Install: `uv sync --all-groups`, then `uv run pre-commit install`
- Lint and format: `uv run ruff check .` and `uv run ruff format --check .`
- Types: `uv run mypy`
- Tests: `uv run pytest`

## Workflow
- Never push to `main`. One branch per task, then open a PR.
- Log design choices as short dated entries in `docs/decisions.md`.

## How we work
- I own the design decisions: event definitions, split, sampling, model choice, evaluation. Claude lays out options and trade-offs, and each decision is logged in `docs/decisions.md`.
- Claude implements to those decisions in small steps and explains each one. I review every change, and I must be able to explain it before it is merged.
- For modeling, go one step at a time and question me on each step before moving on.
- Welcome without asking: reviewing diffs, suggesting tests, fixing CI, docs.
- Explain briefly. Show code only when walking through it or when I ask.

## Code standards
- Type hints on all package code. NumPy-style docstrings.
- No magic numbers: seeds, vintage cutoffs and event definitions live in `configs/default.yaml`.
- Notebooks are named `NN_topic.ipynb`. Outputs are stripped on commit.

## Data and evaluation rules
- Never commit data. Do not redistribute loan-level data or publish extracts (Freddie Mac terms).
- Split by origination vintage, never randomly. Keep all rows of a loan in one split.
- A training loan's outcome must be censored at the date the validation or test period starts.
- Do not use the test set for any choice (features, hyperparameters, calibration).
- Preprocessing is fit on training data only.
- Use only information known at each loan month.
