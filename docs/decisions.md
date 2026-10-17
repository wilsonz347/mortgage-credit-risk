# Decision log

Short, dated entries, newest first. Record what was decided and why.

---

## 2026-10-17: Rate spread and the validation gap

**Built:** FRED download and cache (`macro.py`). Two new loan columns:

| Column | Meaning |
|---|---|
| `market_rate` | FRED 30-year (or 15-year, for terms of 180 months or less) mortgage rate in the origination month |
| `rate_spread` | `orig_rate` minus `market_rate` |

- Origination month = first payment month minus 2.
- Known at origination, so no leakage. Mortgage rates are not revised.

**Tested:** Does the rate spread fix the 30% under-prediction on validation? No (see `notebooks/04_rate_spread_and_gap.ipynb`).

| Features | Cox C-index | Cox predicted / observed |
|---|---|---|
| `orig_rate` (M3) | 0.815 | 0.71 |
| `rate_spread` | 0.812 | 0.61 |
| `rate_spread` + `market_rate` | 0.816 | 0.68 |

**Found the cause:** the 2017 hurricanes.

- Florida (ratio 0.19) and Texas (0.53) hold 41% of validation's 12-month defaults. Every other state is 1.00.
- Most of those loans were in disaster relief, then still 90+ days late after relief ended. Rule B counts those later months.
- Unemployment and house prices improved in 2017 to 2018, so they cannot explain it.

**Decision:** Keep it simple.

- Keep rule B and the M3 features. Report the hurricane effect as a known shock.
- No new default rule, no recalibration, no more economic data for now.
- Keep `market_rate` and `rate_spread` in the tables for later.

---

## 2026-10-17: M3 results and next step

**Results** (5% sample, see `notebooks/03_baselines.ipynb`):

| | 12-month logistic | Cox |
|---|---|---|
| Training defaults used | 169 | 884 |
| Validation AUC (12-month) | 0.800 | 0.812 |
| Validation C-index | 0.805 | 0.815 |
| Validation predicted / observed | 0.67 | 0.71 |
| Test AUC (12-month) | 0.666 | 0.675 |
| Test C-index | 0.70 | 0.72 |
| Test predicted / observed | 0.67 | 0.72 |

**Decision:** Keep Cox as the main baseline.

- It uses all defaults, ranks slightly better (paired bootstrap: wins in 99.8% of resamples), and gives a default curve over time.
- Proportional hazards mostly holds. The owner vs investor ratio may grow with loan age (1.19 to 2.16); watch it.

**Next:** Add economic features from FRED, starting with the rate spread at origination.

- Both models under-predict validation and test by about 30%, and ranking drops on test (COVID).
- Two different models failing the same way points to missing information about the period, not the model type.

**Not now:**

- **CatBoost.** It learns from the same features and period, so it cannot fix the gap.
- **Recalibration.** It would hide the gap instead of explaining it. Revisit before expected loss (M6), since a 30% under-prediction means reserves 30% too low.

---

## 2026-10-16: Baseline models (M3)

**Decision:** Keep the model set small. Start with two baselines, then decide whether to add more.

| Model | Table | Answers |
|---|---|---|
| 12-month logistic regression | `model_loans` | Will the loan default within its first 12 months? |
| Cox proportional hazards | `model_loans` | How does each feature change default risk over time? |

**Why:**

- Simple models first, so any later model has a clear bar to beat.
- More models (CatBoost hazard, competing-risk models) are added only if the baselines show a reason to.

**12-month logistic:**

- Target: default by loan age 12 (`last_age` 12 or less).
- Uses loans watched to age 12 or that defaulted or prepaid sooner. Loans censored earlier are left out.
- 12 months is the industry standard (Basel, IFRS 9) and fits the data. At 24 months, only validation loans that already had an event are usable.
- At 5% this leaves 169 training, 147 validation and 199 test defaults.
- Solver: `newton-cholesky`. The default `lbfgs` stopped after 16 iterations with defaults this rare, and gave wrong coefficients (second homes 0.29 instead of 0.98). Checked against statsmodels.

**Cox:**

- Duration is `months`, event is default.
- Prepayment is treated as censoring. This answers "risk of default while the loan is still open", not the overall chance of default.

**Starting features:** credit score, LTV, DTI, interest rate, loan purpose, occupancy. Add more once these work.

**Library check (Oct 16, 2026, Python 3.12):**

| Library | Works | Note |
|---|---|---|
| scikit-survival 0.28 | Yes | Works with pandas 3.0.6 |
| lifelines 0.30 | Yes | Needs pandas below 3, so it would downgrade pandas to 2.3.3 |

**Library:** scikit-survival, so pandas stays at version 3.

---

## 2026-10-16: Modeling tables

**Decision:** Build two tables from each sample, both in `data/processed/`.

| Table | One row per | Used for |
|---|---|---|
| `model_loan_months_<rate>.parquet` | Loan per month | Discrete-time hazard models |
| `model_loans_<rate>.parquet` | Loan | Cox model, fixed-horizon baselines |

**Event timing:**

- `event` is 0 (none), 1 (default) or 2 (prepay) for that month.
- A loan's rows stop at its first event.
- If a default and a prepayment happen in the same month, default wins.
- Events are labeled first, then the cutoff is applied. An event after the cutoff is removed with its row, so the loan counts as censored.

**Features:**

- Origination fields only for now (`FEATURES` in `pipeline.py`).
- Time-varying features (market rate, ELTV) are left for later. They must use only data known at each month.

---

## 2026-10-15: Default rule, HARP loans and sample rate

### Events

| Event | Rule |
|---|---|
| **Default** | First month the loan is 90+ days late, or the loan ends with a credit loss |
| **Prepayment** | Loan is paid off (zero balance code `01`) |
| **Censored** | Loan is still active, or Freddie Mac sold or removed it (codes `15`, `16`, `96`) |

**Default, in detail** (rule B in `notebooks/01_data_audit.ipynb`):

- 90+ days late means delinquency code `3` or higher, or `RA` (lender took the property).
- Credit loss means zero balance code `02`, `03` or `09`.
- Months in forbearance (`borrower_assistance_plan = F`) or disaster relief
  (`delinquency_due_to_disaster = Y`) do not count as 90+ days late.

**Why rule B:**

- Training ends before COVID, but test loans live through it.
- Counting forbearance months adds a huge 2020 spike that is not real credit loss.
  - 89% of 90+ day late months from Apr 2020 to Dec 2021 were in relief.
  - 95% of loans that reached 90+ days late were current again later.
- 90 days late is also the standard bank definition.

**Why code `01` is all prepayment:**

- Only 1,470 of 5.6 million `01` exits are loans that reached maturity.

### HARP loans

**Decision:** Drop them.

**Why:**

- They are 8.2% of 2015 loans and 4.9% of 2016 loans, but almost none of 2019 loans.
- They default more than twice as often (4.9% vs 1.9%).
- They never have a DTI value.

### Sample

**Decision:** Sample whole loans, not rows.

- 5% while building.
- 10% for final results.

At 10%, this leaves about:

| Split | Defaults |
|---|---|
| Train | 2,200 |
| Validation | 510 |
| Test | 3,100 |

---

## 2026-10-15: Train, validation and test split

**Decision:**

| Split | Loans made in | Months used |
|---|---|---|
| Train | 2015, 2016 | Before Jan 2019 |
| Validation | 2017 | Before Jan 2019 |
| Test | 2019 | Up to Mar 2026 |
| Left out | 2018 | |

**Cutoff:** Jan 2019, when the test loans start. Train and validation never see anything after it.

**Why 2018 is left out:**

- By the cutoff, 2018 loans are less than a year old.
- Very few loans default that early (521 out of 1.29 million).

**Why not the first plan** (train on 2015 to 2017, validate on 2018):

| Plan | Train defaults | Validation defaults |
|---|---|---|
| First plan | about 12,300 | 521 |
| This split | about 22,200 | about 5,100 |

Counts use default rule B on all loans. See `notebooks/01_data_audit.ipynb`.

**Known limit:**

- Training loans are seen up to about 4 years old.
- Test loans are followed for up to 7 years, so later years are a stretch for the model.

---

## 2026-10-15: Project scaffolding (M0)

**Standards:** Copied from `cost-sensitive-targeting`.

- Python 3.12, uv, src layout.
- Ruff (NumPy docstrings), strict mypy, pytest with coverage.
- Pre-commit with nbstripout.
- GitHub Actions CI.

**Dependencies:**

- Start with numpy, pandas, pyarrow and pyyaml only.
- Add model and survival libraries in the milestone that first needs them.

**Data:** Keep all Freddie Mac and FRED files under `data/`, which is gitignored.
