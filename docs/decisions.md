# Decision log

Short, dated entries, newest first. Record what was decided and why.

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

- Uses loans watched for 12 months or that ended sooner. Loans censored earlier are left out.
- 12 months is the industry standard (Basel, IFRS 9) and fits the data. At 24 months, only validation loans that already had an event are usable.
- At 5% this leaves 152 training, 131 validation and 184 test defaults.

**Cox:**

- Duration is `months`, event is default.
- Prepayment is treated as censoring. This answers "risk of default while the loan is still open", not the overall chance of default.

**Starting features:** credit score, LTV, DTI, interest rate, loan purpose, occupancy. Add more once these work.

**Library check (Oct 16, 2026, Python 3.12):**

| Library | Works | Note |
|---|---|---|
| scikit-survival 0.28 | Yes | Works with pandas 3.0.6 |
| lifelines 0.30 | Yes | Needs pandas below 3, so it would downgrade pandas to 2.3.3 |

Library choice for Cox: to be decided.

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

**Who writes what:** From Oct 16, 2026, Claude writes the data side and the user writes the modeling side. See `CLAUDE.md`.

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
