"""Tests for reading raw files and sampling loans."""

from dataclasses import replace
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from mortgage_credit_risk.config import Config
from mortgage_credit_risk.data import (
    build_sample,
    in_sample_sql,
    raw_files,
    read_header,
)


def test_read_header_returns_snake_case_names(raw_dir: Path) -> None:
    orig = read_header(raw_dir, "orig")
    perf = read_header(raw_dir, "perf")

    assert len(orig) == 31
    assert len(perf) == 35
    assert orig[0] == "classic_fico"
    assert "original_debt_to_income_dti_ratio" in orig
    assert perf[:2] == ["loan_identifier", "period"]


def test_raw_files_missing_year_raises(raw_dir: Path) -> None:
    with pytest.raises(FileNotFoundError, match="2020"):
        raw_files(raw_dir, "orig", (2015, 2020))


def test_build_sample_keeps_split_vintages_and_drops_harp(cfg: Config) -> None:
    loans_path, _ = build_sample(cfg, rate=1.0)
    loans = pd.read_parquet(loans_path)

    # 2018 is not in any split, and F15Q10000002 is a HARP loan.
    assert loans["loan_id"].tolist() == [
        "F15Q10000001",
        "F15Q10000003",
        "F16Q20000001",
        "F17Q30000001",
        "F19Q40000001",
    ]
    assert loans["vintage"].tolist() == [2015, 2015, 2016, 2017, 2019]
    assert loans["orig_quarter"].iloc[0] == "2015Q1"


def test_build_sample_keeps_harp_when_configured(cfg: Config) -> None:
    loans_path, _ = build_sample(replace(cfg, drop_harp=False), rate=1.0)

    assert "F15Q10000002" in set(pd.read_parquet(loans_path)["loan_id"])


def test_missing_codes_become_null(cfg: Config) -> None:
    loans_path, _ = build_sample(cfg, rate=1.0)
    loans = pd.read_parquet(loans_path).set_index("loan_id")

    assert pd.isna(loans.loc["F15Q10000003", "credit_score"])
    assert pd.isna(loans.loc["F15Q10000003", "dti"])
    assert pd.isna(loans.loc["F15Q10000003", "occupancy"])
    assert loans.loc["F15Q10000001", "credit_score"] == 760
    assert loans.loc["F15Q10000001", "dti"] == 35


def test_loan_months_match_sampled_loans(cfg: Config) -> None:
    loans_path, months_path = build_sample(cfg, rate=1.0)
    loans = pd.read_parquet(loans_path)
    months = pd.read_parquet(months_path)

    assert set(months["loan_id"]) == set(loans["loan_id"])
    assert len(months) == 8
    assert months["period"].dtype == "int32"
    assert months.set_index(["loan_id", "period"]).index.is_monotonic_increasing
    reo = months.query("loan_id == 'F15Q10000003' and period == 201503")
    assert reo["delinquency"].item() == "RA"


def sample_ids(seed: int, rate: float) -> set[str]:
    ids = "SELECT 'F' || CAST(range AS VARCHAR) AS loan_id FROM range(20000)"
    sql = f"SELECT loan_id FROM ({ids}) WHERE {in_sample_sql('loan_id', seed, rate)}"
    return {row[0] for row in duckdb.sql(sql).fetchall()}


def test_sample_rate_is_close_to_requested() -> None:
    assert len(sample_ids(seed=42, rate=0.1)) / 20000 == pytest.approx(0.1, abs=0.01)


def test_sample_is_stable_and_nested() -> None:
    small = sample_ids(seed=42, rate=0.05)

    assert small == sample_ids(seed=42, rate=0.05)
    assert small < sample_ids(seed=42, rate=0.10)
    assert small != sample_ids(seed=7, rate=0.05)
