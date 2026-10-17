"""Tests for loss given default and expected loss."""

from pathlib import Path

import pandas as pd
import pytest

from mortgage_credit_risk.loss import (
    expected_loss,
    loan_outcomes,
    loss_given_default,
    outcome_status,
)


@pytest.fixture
def months_path(tmp_path: Path) -> Path:
    rows = [
        # loan, period, delinquency, zero balance code, actual loss
        ("lost", 201801, "03", None, None),
        ("lost", 201806, "RA", "09", 30_000.0),
        ("cured", 201801, "03", None, None),
        ("cured", 201802, "00", None, None),
        ("still_late", 201801, "03", None, None),
        ("still_late", 201812, "08", None, None),
        ("later_loss", 201801, "03", None, None),
        ("later_loss", 201905, "RA", "09", 50_000.0),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "loan_id",
            "period",
            "delinquency",
            "zero_balance_code",
            "actual_loss",
        ],
    )
    path = tmp_path / "months.parquet"
    df.to_parquet(path)
    return path


def test_loan_outcomes_only_use_months_before_cutoff(months_path: Path) -> None:
    ids = pd.Series(["lost", "cured", "still_late", "later_loss"])
    out = loan_outcomes(months_path, ids, before=201901)

    assert out.loc["lost", "loss"] == 30_000
    assert out.loc["later_loss", "loss"] == 0
    assert pd.isna(out.loc["later_loss", "zero_balance_code"])
    assert out.loc["cured", "last_delinquency"] == "00"


def test_outcome_status_and_all_months(months_path: Path) -> None:
    ids = pd.Series(["lost", "cured", "still_late", "later_loss"])

    before = outcome_status(loan_outcomes(months_path, ids, before=201901))
    after = outcome_status(loan_outcomes(months_path, ids))

    assert before.to_dict() == {
        "cured": "current again",
        "later_loss": "still late",
        "lost": "credit loss",
        "still_late": "still late",
    }
    assert after["later_loss"] == "credit loss"


def test_lgd_uses_only_known_outcomes(months_path: Path) -> None:
    defaults = pd.DataFrame(
        {
            "loan_id": ["lost", "cured", "still_late", "later_loss"],
            "orig_upb": [100_000.0, 100_000.0, 100_000.0, 100_000.0],
        }
    )
    outcomes = loan_outcomes(months_path, defaults["loan_id"], before=201901)

    # Known: "lost" (30k) and "cured" (0). Still-late loans are left out.
    assert loss_given_default(defaults, outcomes) == pytest.approx(0.15)


def test_expected_loss_is_the_product() -> None:
    out = expected_loss(pd.Series([0.01, 0.02]), 0.1, pd.Series([200_000, 100_000]))

    assert out.tolist() == pytest.approx([200.0, 200.0])
