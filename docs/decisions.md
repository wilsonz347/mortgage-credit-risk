# Decision log

Short, dated entries, newest first. Record what was decided and why.

## 2026-10-15: Default rule, HARP loans and sample rate

- **Default:** A loan defaults in the first month it is 90+ days late (delinquency code 3 or more, or `RA`), or when it ends with a credit loss (zero balance code 02, 03 or 09). Months in forbearance (`borrower_assistance_plan = F`) or disaster relief (`delinquency_due_to_disaster = Y`) do not count. This is rule B in `notebooks/01_data_audit.ipynb`.
- **Why rule B:** Training ends before COVID, but test loans live through it. Counting forbearance months would add a huge 2020 spike that is not real credit loss (89% of 90+ day late months in Apr 2020 to Dec 2021 were in relief, and 95% of loans that reached 90+ days late were current again later). 90 days late is also the standard bank definition.
- **Prepayment:** Zero balance code 01. Only 1,470 of 5.6 million of these are loans that reached maturity, so treat all of them as prepayment.
- **Censored:** Loans still active, and loans sold or removed by Freddie Mac (codes 15, 16, 96).
- **HARP loans are dropped.** They are 8.2% of 2015 loans and 4.9% of 2016 loans but almost none of 2019 loans, default more than twice as often (4.9% vs 1.9%), and never have DTI.
- **Sample:** Sample whole loans, not rows. Use 5% while building and 10% for final results. At 10% this leaves about 2,200 training, 510 validation and 3,100 test defaults.

## 2026-10-15: Train, validation and test split

- **Split:** Train on 2015 and 2016 loans. Validate on 2017 loans. Test on 2019 loans.
- **Cutoff:** Jan 2019, when the test loans start. Train and validation only use months before it. Test loans are followed to Mar 2026.
- **2018 loans are left out.** By the cutoff they are less than a year old, and very few loans default that early (521 out of 1.29 million).
- **Why not the first plan (train 2015 to 2017, validate 2018):** It left only about 12,300 training defaults and 521 validation defaults. This split gives about 22,200 and 5,100 (default rule B, all loans, see `notebooks/01_data_audit.ipynb`).
- **Known limit:** Training loans are seen up to about 4 years old. Test loans are followed for up to 7 years, so later years are a stretch for the model.

## 2026-10-15: Project scaffolding (M0)

- **Standards:** Copy the setup from `cost-sensitive-targeting`: Python 3.12, uv, src layout, ruff (numpy docstrings), strict mypy, pytest with coverage, pre-commit with nbstripout, GitHub Actions CI.
- **Dependencies:** Start with numpy, pandas, pyarrow and pyyaml only. Add model and survival libraries in the milestone that first needs them.
- **Data:** Keep all Freddie Mac and FRED files under `data/`, which is gitignored.
