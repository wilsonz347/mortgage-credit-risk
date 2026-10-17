"""Tests for the vintage split and calendar cutoff."""

import duckdb
import pandas as pd

from mortgage_credit_risk.config import load_config
from mortgage_credit_risk.events import DEFAULT, NO_EVENT, label_sql
from mortgage_credit_risk.splits import TEST, TRAIN, VALIDATION, split_sql

CUTOFF = load_config().split.cutoff_period


def months(loan_id: str, vintage: int, start: int, n: int) -> list[dict[str, object]]:
    """N monthly rows from ``start`` (YYYYMM), all current, no exit."""
    rows = []
    year, month = divmod(start, 100)
    for _ in range(n):
        rows.append(
            {
                "loan_id": loan_id,
                "vintage": vintage,
                "period": year * 100 + month,
                "delinquency": "00",
                "assistance_plan": None,
                "disaster_flag": None,
                "zero_balance_code": None,
            }
        )
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return rows


def run(df: pd.DataFrame) -> pd.DataFrame:
    cfg = load_config()
    sql = split_sql(f"({label_sql('df', cfg.events)})", cfg.split)
    return duckdb.sql(sql).df()


def test_vintages_map_to_splits_and_other_years_are_dropped() -> None:
    df = pd.DataFrame(
        months("a", 2015, 201501, 3)
        + months("b", 2017, 201701, 3)
        + months("c", 2018, 201801, 3)
        + months("d", 2019, 201901, 3)
    )
    out = run(df).groupby("loan_id")["split"].unique().to_dict()

    assert {k: list(v) for k, v in out.items()} == {
        "a": [TRAIN],
        "b": [VALIDATION],
        "d": [TEST],
    }


def test_train_and_validation_stop_before_cutoff() -> None:
    df = pd.DataFrame(months("a", 2016, 201801, 24) + months("b", 2017, 201801, 24))
    out = run(df)

    assert out["period"].max() < CUTOFF
    assert len(out) == 2 * 12


def test_test_rows_after_cutoff_are_kept() -> None:
    out = run(pd.DataFrame(months("a", 2019, 201901, 30)))

    assert len(out) == 30


def test_default_after_cutoff_is_censored_in_training() -> None:
    rows = months("a", 2016, 201811, 6)
    rows[4]["delinquency"] = "03"  # Mar 2019, after the cutoff
    out = run(pd.DataFrame(rows))

    assert out["period"].tolist() == [201811, 201812]
    assert out["event"].tolist() == [NO_EVENT, NO_EVENT]


def test_default_before_cutoff_is_kept_in_training() -> None:
    rows = months("a", 2016, 201810, 6)
    rows[1]["delinquency"] = "03"  # Nov 2018
    out = run(pd.DataFrame(rows))

    assert out["event"].tolist() == [NO_EVENT, DEFAULT]
