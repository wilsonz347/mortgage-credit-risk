# mortgage-credit-risk

Survival models for mortgage default and prepayment risk, using the Freddie Mac Single-Family Loan-Level Dataset. Predicts the monthly risk of default and of prepayment over a loan's life, validates on later loan vintages, and turns the results into calibrated expected-loss estimates with stress and monitoring checks.

**Status:** scaffolding done (M0). See [docs/project_spec.md](docs/project_spec.md).

## Planned approach

- Competing-risk outcomes (default, prepayment) with censoring.
- Baselines: logistic regression and Cox proportional hazards. Main model: discrete-time hazard with CatBoost.
- Validation by origination vintage, with outcomes censored at the start of the evaluation period.
- Calibration by risk band and vintage, expected loss, macro stress scenario, and drift monitoring.

## Setup

```bash
uv sync --all-groups
uv run pre-commit install
uv run pytest
```

## Data

The data is not included in this repository and must not be redistributed. Register and download it from Freddie Mac's [Single-Family Loan-Level Dataset](https://freddiemac.com/research/datasets/sf-loanlevel-dataset) page, then place files under `data/` (gitignored).

## License

MIT
