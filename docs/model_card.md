# Model card: Cox default model

## What it is

A Cox proportional hazards model that predicts each mortgage's monthly risk of default over its life. Default means 90+ days late (not counting COVID forbearance or disaster relief months) or a credit-loss exit. Prepayment is treated as censoring.

- **Inputs** (all known when the loan is made): credit score, LTV, DTI, interest rate, loan purpose, occupancy.
- **Output:** a survival curve per loan, from which the chance of default by any age (for example 12 or 36 months) follows.
- **Library:** scikit-survival. Settings in `configs/default.yaml`.

## Intended use

- Ranking new fixed-rate Freddie Mac loans by default risk.
- Estimating how many loans in a group will default over the first few years.
- Learning and demonstration. This is a portfolio project, not a production model.

## Not for

- Decisions about individual borrowers.
- Loans unlike the training data: non-conforming, adjustable-rate, or very high LTV or DTI.
- Loss forecasts in a housing market very different from 2015 to 2018 without a new loss rate.

## Data

- Train: 146,446 loans made in 2015 and 2016, 884 defaults, followed to Dec 2018.
- Validation: 66,288 loans made in 2017, followed to Dec 2018.
- Test: 88,906 loans made in 2019, followed to Mar 2026.
- A 5% random sample of Freddie Mac loans. HARP refinance loans are excluded.

## Performance

| | Validation | Test |
|---|---|---|
| C-index | 0.815 | 0.72 |
| AUC, default within 12 months | 0.812 | 0.675 |
| Predicted / observed defaults, 12 months | 0.71 | 0.72 |
| Predicted / observed defaults, 36 months | | 1.07 |

## Limits

- **Shocks after origination are invisible.** The model under-predicted 2017 loans by about 30%, almost all from Hurricanes Irma and Harvey in Florida and Texas. Its ranking dropped on 2019 loans because of COVID.
- **Only the first 4 years are learned.** Training loans are seen up to 47 months old. Risk after that is an extrapolation.
- **Proportional hazards mostly holds,** but the owner vs investor effect may grow with loan age (1.2 early, 2.2 later).
- **Expected loss uses one fixed loss rate** (about 1% of balance). On 2019 loans it over-predicted losses by about 2x, because house prices rose sharply in 2021 and 2022.
- **Prepayment is treated as censoring,** so long-horizon default chances are slightly too high.

## Fairness

No protected characteristics (race, sex, age) are in the data or the model. State and occupancy are used and can be related to protected characteristics. The model has not been tested for fair lending.

## Version

v0.1.1, Oct 2026. Details of every choice are in [decisions.md](decisions.md).
