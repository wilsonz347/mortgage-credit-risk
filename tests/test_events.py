"""Tests for event labels, using hand-built loan histories."""

from dataclasses import replace

import duckdb
import pandas as pd
import pytest

from mortgage_credit_risk.config import Events, load_config
from mortgage_credit_risk.events import DEFAULT, NO_EVENT, PREPAY, label_sql


@pytest.fixture
def events() -> Events:
    return load_config().events


def history(loan_id: str, *months: tuple[str, str, str, str]) -> pd.DataFrame:
    """Build one loan's history from (delinquency, plan, disaster, zero balance)."""
    rows = [
        {
            "loan_id": loan_id,
            "period": 201601 + i,
            "delinquency": d,
            "assistance_plan": plan or None,
            "disaster_flag": disaster or None,
            "zero_balance_code": zbc or None,
        }
        for i, (d, plan, disaster, zbc) in enumerate(months)
    ]
    return pd.DataFrame(rows)


def label(df: pd.DataFrame, events: Events) -> list[int]:
    out = duckdb.sql(f"SELECT * FROM ({label_sql('df', events)}) ORDER BY period")
    return [int(e) for e in out.df()["event"]]


def by_loan(df: pd.DataFrame, events: Events) -> pd.DataFrame:
    return duckdb.sql(f"SELECT loan_id, event FROM ({label_sql('df', events)})").df()


def test_first_90_day_late_month_is_default(events: Events) -> None:
    df = history(
        "a",
        ("00", "", "", ""),
        ("01", "", "", ""),
        ("02", "", "", ""),
        ("03", "", "", ""),
        ("04", "", "", ""),
    )

    assert label(df, events) == [NO_EVENT, NO_EVENT, NO_EVENT, DEFAULT]


def test_default_is_dated_to_first_time_late(events: Events) -> None:
    df = history("a", ("03", "", "", ""), ("00", "", "", ""), ("03", "", "", ""))

    assert label(df, events) == [DEFAULT]


def test_reo_counts_as_default(events: Events) -> None:
    df = history("a", ("00", "", "", ""), ("RA", "", "", ""))

    assert label(df, events) == [NO_EVENT, DEFAULT]


def test_late_in_forbearance_is_not_default(events: Events) -> None:
    df = history("a", ("03", "F", "", ""), ("04", "F", "", ""), ("00", "", "", ""))

    assert label(df, events) == [NO_EVENT, NO_EVENT, NO_EVENT]


def test_late_in_disaster_relief_is_not_default(events: Events) -> None:
    df = history("a", ("03", "", "Y", ""), ("00", "", "", ""))

    assert label(df, events) == [NO_EVENT, NO_EVENT]


def test_late_after_forbearance_ends_is_default(events: Events) -> None:
    df = history("a", ("03", "F", "", ""), ("04", "", "", ""), ("05", "", "", ""))

    assert label(df, events) == [NO_EVENT, DEFAULT]


def test_relief_counts_when_not_ignored(events: Events) -> None:
    df = history("a", ("00", "", "", ""), ("03", "F", "", ""))

    assert label(df, replace(events, ignore_relief_months=False)) == [
        NO_EVENT,
        DEFAULT,
    ]


@pytest.mark.parametrize("code", ["02", "03", "09"])
def test_credit_loss_exit_is_default(events: Events, code: str) -> None:
    df = history("a", ("00", "", "", ""), ("00", "", "", code))

    assert label(df, events) == [NO_EVENT, DEFAULT]


def test_payoff_is_prepay(events: Events) -> None:
    df = history("a", ("00", "", "", ""), ("00", "", "", "01"))

    assert label(df, events) == [NO_EVENT, PREPAY]


@pytest.mark.parametrize("code", ["15", "16", "96"])
def test_sold_or_removed_is_censored(events: Events, code: str) -> None:
    df = history("a", ("00", "", "", ""), ("02", "", "", code))

    assert label(df, events) == [NO_EVENT, NO_EVENT]


def test_default_before_payoff_wins(events: Events) -> None:
    df = history("a", ("03", "", "", ""), ("00", "", "", ""), ("00", "", "", "01"))

    assert label(df, events) == [DEFAULT]


def test_loans_are_labeled_separately(events: Events) -> None:
    df = pd.concat(
        [
            history("a", ("00", "", "", ""), ("03", "", "", "")),
            history("b", ("00", "", "", ""), ("00", "", "", ""), ("00", "", "", "01")),
        ]
    )
    out = by_loan(df, events)

    assert out.groupby("loan_id")["event"].apply(list).to_dict() == {
        "a": [NO_EVENT, DEFAULT],
        "b": [NO_EVENT, NO_EVENT, PREPAY],
    }
