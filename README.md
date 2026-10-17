# mortgage-credit-risk

Predicts when mortgages default, and how much money that costs, using Freddie Mac loan data. Models are trained on older loans and tested on newer ones, the way a lender would use them.

## Problem

A lender needs to know which loans will default, and when, to set aside money for losses. A yes/no default flag ignores timing and loans that end early (paid off, or still running when the data ends). This project uses survival models, which handle both.

## Data

- **Freddie Mac Single-Family Loan-Level Dataset:** loans made in 2015 to 2019, with monthly payment history to Mar 2026. A 5% random sample of loans (about 300,000 loans and 17 million loan-months; 9 million loan-months remain after the cutoff and each loan's first event).
- **FRED** (Federal Reserve Economic Data): market mortgage rates, unemployment, house prices.

Data is not included. See [How to run](#how-to-run).

## Approach

**Default:** the first month a loan is 90+ days late, or it ends in a credit loss. Months in COVID forbearance or disaster relief do not count. Paid-off loans are prepayments. Loans still running are censored.

**Split by when loans were made, with a cutoff date:**

| Split | Loans made in | Months used |
|---|---|---|
| Train | 2015, 2016 | Before Jan 2019 |
| Validation | 2017 | Before Jan 2019 |
| Test | 2019 | To Mar 2026 |

Training never sees anything after Jan 2019, so the test is like using the model in real time.

**Models:**

1. Logistic regression: default within 12 months.
2. Cox proportional hazards: monthly default risk over the life of the loan. This is the main model.

**Expected loss:** chance of default within 36 months (Cox) x loss rate (from training) x loan balance.

## Results

| | 12-month logistic | Cox |
|---|---|---|
| Validation AUC (12-month) | 0.800 | 0.812 |
| Validation C-index | 0.805 | 0.815 |
| Test AUC (12-month) | 0.666 | 0.675 |
| Test C-index | 0.70 | 0.72 |

**Expected loss on test** (2019 loans, first 36 months):

| | Predicted | Actual |
|---|---|---|
| Defaults | 957 | 898 |
| Dollars lost | $2.25 million | $0.95 million |

## Key findings

1. **Credit score matters most.** About 44 points higher roughly halves the monthly default risk.
2. **Survival models use the data better.** Cox learns from all 884 training defaults; the 12-month model only from 169. Cox finds clear DTI and LTV effects that the 12-month window hides.
3. **The 2017 under-prediction was hurricanes.** Both models predicted about 30% too few defaults on 2017 loans. Almost all of it was Florida and Texas after Hurricanes Irma and Harvey. Every other state was predicted correctly.
4. **Ranking dropped on 2019 loans because of COVID.** Who got into trouble in 2020 depended on jobs and industry, which loan data at origination cannot show.
5. **Default counts were right, losses were not.** Over 36 months the model predicted 957 defaults and 898 happened. Losses were over-predicted by about 2x, because house prices rose 17% in 2021 and 15% in 2022, so defaulted homes sold with little loss.

Details are in the notebooks and in [docs/decisions.md](docs/decisions.md). Limits are in the [model card](docs/model_card.md).

## Repository

```
configs/default.yaml        all settings: split, event codes, features, sample rates
src/mortgage_credit_risk/
    data.py                 read raw files, sample loans
    events.py               default / prepayment labels
    splits.py               train / validation / test and the cutoff
    pipeline.py             build the modeling tables
    macro.py                FRED download
    models.py               logistic and Cox
    evaluate.py             AUC, C-index, calibration, bootstrap
    loss.py                 expected loss
notebooks/
    01_data_audit           data checks, default definition
    02_modeling_data        the modeling tables
    03_baselines            logistic vs Cox
    04_rate_spread_and_gap  the validation gap (hurricanes)
    05_expected_loss        dollars
docs/                       decisions, model card
tests/                      tests, including leakage checks on hand-built loans
```

## How to run

1. **Install:**

   ```bash
   uv sync --all-groups
   uv run pre-commit install
   ```

2. **Get the data.** Register at Freddie Mac's [Single-Family Loan-Level Dataset](https://freddiemac.com/research/datasets/sf-loanlevel-dataset) page and download the 2015 to 2019 historical files and the header files into `data/raw/`:

   ```
   data/raw/header/origination_data_file_header.txt
   data/raw/header/performance_data_file_header.txt
   data/raw/historical_data_2015/historical_data_2015Q1/orig_2015Q1.txt
   ...
   ```

3. **FRED key.** Get a free key at [fred.stlouisfed.org](https://fred.stlouisfed.org) and add `FRED_API_KEY=...` to `.env`.

4. **Build the modeling tables** (about a minute):

   ```bash
   uv run python -m mortgage_credit_risk.pipeline
   ```

5. **Run the notebooks** in order, and the tests with `uv run pytest`.

## Data terms

Freddie Mac data is free for non-commercial use and must not be redistributed. No loan-level data is in this repository.

## License

MIT
