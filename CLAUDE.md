# mortgage-credit-risk

Survival models for mortgage default and prepayment risk (Freddie Mac Single-Family Loan-Level Dataset). Full plan: `docs/project_spec.md`. Read it before suggesting design changes.

Status: M1 (data audit) done. Next: M2 loader, sampling and splits.

## Commands
- Install: `uv sync --all-groups`, then `uv run pre-commit install`
- Lint and format: `uv run ruff check .` and `uv run ruff format --check .`
- Types: `uv run mypy`
- Tests: `uv run pytest`

## Workflow
- Never push to `main`. One branch per task, then open a PR.
- Log design choices as short dated entries in `docs/decisions.md`.

## Who writes what
- Claude writes the data side: loading, cleaning, event definitions, splits, modeling tables, data analysis notebooks.
- I write the modeling side: hazard setup, models, metrics, evaluation. Do not write or edit modeling code unless I ask.
- Welcome without asking: reviewing my diffs, suggesting tests, fixing CI, docs.
- Explain briefly. Do not paste code unless I ask for it.

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
