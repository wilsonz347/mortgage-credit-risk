"""End-to-end tests for building the modeling tables from raw files."""

from dataclasses import replace

import pandas as pd
import pytest

from mortgage_credit_risk.config import Config
from mortgage_credit_risk.events import DEFAULT, NO_EVENT, PREPAY
from mortgage_credit_risk.pipeline import FEATURES, build_model_tables, main


def test_loan_table_has_one_row_per_loan_with_outcome(cfg: Config) -> None:
    _, loan_path = build_model_tables(cfg, rate=1.0)
    loans = pd.read_parquet(loan_path).set_index("loan_id")

    # Fixture: 0001 pays off in its third month, 0003 reaches REO in its second.
    assert loans.loc["F15Q10000001", ["months", "event", "event_period"]].tolist() == [
        3,
        PREPAY,
        201504,
    ]
    assert loans.loc["F15Q10000003", ["months", "event"]].tolist() == [2, DEFAULT]
    assert loans.loc["F16Q20000001", "event"] == NO_EVENT
    assert pd.isna(loans.loc["F16Q20000001", "event_period"])
    assert loans["split"].to_dict() == {
        "F15Q10000001": "train",
        "F15Q10000003": "train",
        "F16Q20000001": "train",
        "F17Q30000001": "validation",
        "F19Q40000001": "test",
    }


def test_loan_month_table_matches_loan_table(cfg: Config) -> None:
    hazard_path, loan_path = build_model_tables(cfg, rate=1.0)
    hazard = pd.read_parquet(hazard_path)
    loans = pd.read_parquet(loan_path).set_index("loan_id")

    assert set(FEATURES) <= set(hazard.columns)
    assert hazard.groupby("loan_id").size().to_dict() == loans["months"].to_dict()
    assert (
        hazard.groupby("loan_id")["event"].max().to_dict() == loans["event"].to_dict()
    )


def test_test_rows_before_cutoff_raise(cfg: Config) -> None:
    # Fixture's 2019 loan has a row in Nov 2019, before this later cutoff.
    late_cutoff = replace(cfg, split=replace(cfg.split, cutoff_period=202001))

    with pytest.raises(ValueError, match="before the cutoff"):
        build_model_tables(late_cutoff, rate=1.0)


def test_main_builds_tables(
    cfg: Config, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("mortgage_credit_risk.pipeline.load_config", lambda _: cfg)
    main([])

    assert "model_loans_5pct.parquet" in capsys.readouterr().out


def test_rate_spread_uses_market_rate_in_origination_month(cfg: Config) -> None:
    _, loan_path = build_model_tables(cfg, rate=1.0)
    loans = pd.read_parquet(loan_path).set_index("loan_id")

    # Fixture loans pay first in Mar 2015, so they originated in Jan 2015,
    # when the fake 30-year rate was 3.8%. Their rate is 4.0%.
    assert loans.loc["F15Q10000001", "market_rate"] == pytest.approx(3.8)
    assert loans.loc["F15Q10000001", "rate_spread"] == pytest.approx(0.2)
