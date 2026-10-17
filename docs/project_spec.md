# Project Spec: Mortgage Credit Risk (survival models)

*Last updated: October 15, 2026. Written so it can be pasted into a new chat to restore context.*

---

## 0. Context (paste into a new chat)

- **Who I am:** Third-year Data Science student at Boston University. Intermediate data science. Python, SQL, pandas, scikit-learn, CatBoost, PySpark/Databricks.
- **Goal:** A second production-level applied DS portfolio project, in finance, where I learn a lot.
- **How I work:** I write the core code myself. Claude gives code only when I ask. Claude guides architecture, decisions and workflow, and reviews my PRs. Keep responses concise.
- **Standards:** Same as my first project (`cost-sensitive-targeting`): `uv`, src layout, ruff, mypy, pytest, CI, NumPy docstrings, PRs to `main`, `CLAUDE.md`, a dated `docs/decisions.md`.
- **Status:** Planning. No code yet.

---

## 1. Problem

A lender needs to know which mortgages will default or prepay, and *when*, to estimate expected losses and set reserves. A yes/no default flag ignores timing and ignores loans that leave the portfolio early. This project models the full loan lifecycle.

**One-sentence version:** Predict, for each newly originated mortgage, the monthly risk of default and of prepayment over its life, validate it on later loan vintages, and turn it into a calibrated expected-loss estimate with stress and monitoring checks.

---

## 2. Data

**Source:** Freddie Mac Single-Family Loan-Level Dataset (SFLLD). Verified from the [dataset page](https://freddiemac.com/research/datasets/sf-loanlevel-dataset) on Oct 15, 2026:
- Loan-level credit performance for mortgages Freddie Mac purchased or guaranteed.
- Standard and Non-Standard datasets cover about 56 million mortgages originated from Jan 1, 1999 to Mar 31, 2026. Monthly performance runs through Mar 31, 2026.
- Fields include prepayments, foreclosure alternatives, REOs, and loss items (net sale proceeds, MI recoveries, expenses).
- **Access:** free registration and sign-in required (Clarity Data Intelligence). Free for non-commercial and academic/research use. Commercial redistribution needs a license. Data is unaudited.
- **Do not commit the data to Git.** Do not redistribute it.

**Not verified yet (check before building):**
- File format, size, and the exact origination and performance field list. The page points to a separate file layout spreadsheet and user guide.
- Whether sample files are enough for early development (the download page lists sample files).
- The page's text and metadata disagree on the data's end year, so confirm what you actually download.

**Second source:** FRED (free API key) for macro series: unemployment, mortgage rates, house price index. Not verified yet.

---

## 3. Modeling design

### Outcome definition (decide first, log in `docs/decisions.md`)
- **Events:** default (define it: e.g., 90+ days delinquent or foreclosure-related termination) and voluntary prepayment. These compete: a loan can only end one way.
- **Censoring:** loans still active at the end of the observation window.
- **Time scale:** months since origination.

### Features
- **At origination:** credit score, loan-to-value, debt-to-income, interest rate, loan purpose, property type, occupancy, state, origination channel.
- **Time-varying (stretch):** current interest rate vs. market rate, unemployment, house price change. Only use information known at that month.

### Models (simple to complex)
1. **Baseline:** logistic regression on "default within 12 months."
2. **Survival baseline:** Cox proportional hazards (or discrete-time logistic hazard).
3. **Main model:** discrete-time hazard with CatBoost on loan-month records, one model per event.
4. **Check:** compare against the simple baselines. Report whether CatBoost earns its complexity.

### Splits and leakage (the core of the project)
- **Split by origination vintage,** not randomly. For example: train on early vintages, validate on a later one, test on the latest.
- **Calendar leakage:** a training loan's outcome must be censored at the date the validation or test period starts. Using later performance data for training labels leaks the future.
- **Same loan, many rows:** keep all of a loan's rows in one split.
- **Sampling:** loan-month data gets huge. Sample loans (not rows), and record the sampling rule.
- **Special periods:** 2008 and 2020 behave differently. Treat them deliberately, not by accident.

### Evaluation
- **Discrimination:** AUC and time-dependent concordance on later vintages.
- **Calibration:** predicted vs. observed default rates by risk band and by vintage. This matters more than AUC for expected loss.
- **Cumulative default and prepayment curves:** predicted vs. observed over time.
- **Expected loss:** probability of default × loss severity × balance. MVP uses observed average severity by segment. Modeling severity is a stretch goal.
- **Stress test:** rerun with a recession-style macro scenario and report how the portfolio loss changes. State the limits plainly.
- **Monitoring:** population stability index on key features and scores across vintages, plus calibration drift.
- **Uncertainty:** bootstrap over loans for confidence intervals.

---

## 4. Proposed stack (minimal)

Python 3.12, `uv`, pandas, pyarrow (Parquet), scikit-learn, CatBoost, a survival library (check `lifelines` or `scikit-survival` compatibility before choosing), statsmodels, matplotlib, pytest, ruff, mypy, MLflow (optional, later). DuckDB or PySpark only if memory forces it. No dashboards, no cloud infrastructure.

---

## 5. Repo layout (same standards as `cost-sensitive-targeting`)

```
mortgage-credit-risk/
├── pyproject.toml  uv.lock  README.md  LICENSE  CLAUDE.md
├── configs/default.yaml        # seeds, vintage cutoffs, event definitions
├── data/                       # gitignored
├── docs/                       # decisions.md, data_card.md, model_card.md
├── notebooks/                  # NN_topic.ipynb, outputs stripped
├── src/mortgage_credit_risk/   # data, splits, features, models, evaluate, stress, monitor
└── tests/                      # includes a small committed fixture
```

---

## 6. Milestones

| Milestone | Deliverable |
|---|---|
| M0 | Scaffolding: copy the standards from the first project. CI green. |
| M1 | Data audit: field list, event definitions, data card. |
| M2 | Loader, sampling, vintage splits (with leakage tests). |
| M3 | Baselines: logistic and Cox. |
| M4 | CatBoost hazard model, competing risks. |
| M5 | Calibration, curves, vintage-by-vintage evaluation. |
| M6 | Expected loss, macro stress test, monitoring report. |
| M7 | README, model card, tag `v0.1.0`. |

**Time estimate:** about 8–10 weeks at 8–10 hours a week. This is my guess, not measured. A usable result (M5) should exist around week 5–6.

---

## 7. Working agreement with Claude

- I own the design decisions: event definitions, split, sampling, model choice, evaluation.
- Claude lays out options and trade-offs, then implements to my decisions in small, explained steps.
- I review every change and get a step-by-step explanation before accepting it.
- Write down what I expect each result to look like before running it.
- Hand-check at least one calculation per module against a tiny example.

---

## 8. Risks

- Data is large. Mitigate by sampling loans and using sample files first.
- Event and censoring definitions are easy to get wrong. Mitigate with tests on hand-built loans.
- Calibration may drift across vintages. That is a finding, not a failure.
- Macro stress results are scenario-based, not forecasts. Say so.
- Terms of use: do not redistribute data or publish loan-level extracts.

---

## 9. Open questions

- Which default definition to use?
- Which vintages for train, validation and test?
- Does CatBoost beat the Cox model on calibration, not just AUC?
- Whether to model loss severity or use averages.

---

## 10. First actions

1. Register for the data and download the sample files plus the file layout and user guide.
2. Create the repo from the standards of the first project (M0).
3. Write the data card and event definitions before any modeling.
