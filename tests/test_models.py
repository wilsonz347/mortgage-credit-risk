"""Tests for the baseline models."""

import numpy as np
import pandas as pd
import pytest

from mortgage_credit_risk.config import load_config
from mortgage_credit_risk.events import DEFAULT, NO_EVENT, PREPAY
from mortgage_credit_risk.models import (
    coefficients,
    default_by,
    fit_cox,
    fit_logistic,
    hazard_ratios,
    horizon_label,
    preprocessor,
)
from tests.simulate import simulated_survival


def test_horizon_label_keeps_known_loans_and_sets_target() -> None:
    loans = pd.DataFrame(
        {
            "loan_id": [
                "early_default",
                "late_default",
                "early_prepay",
                "watched",
                "too_short",
            ],
            "last_age": [5, 20, 3, 30, 8],
            "event": [DEFAULT, DEFAULT, PREPAY, NO_EVENT, NO_EVENT],
        }
    )
    out = horizon_label(loans, horizon=12).set_index("loan_id")["target"]

    assert out.to_dict() == {
        "early_default": 1,
        "late_default": 0,
        "early_prepay": 0,
        "watched": 0,
    }


def test_default_exactly_at_horizon_counts() -> None:
    loans = pd.DataFrame({"last_age": [12], "event": [DEFAULT]})

    assert horizon_label(loans, horizon=12)["target"].tolist() == [1]


def test_preprocessor_learns_only_from_training_data() -> None:
    train = pd.DataFrame({"x": [1.0, 2.0, np.nan, 5.0], "c": ["a", "b", "a", "b"]})
    other = pd.DataFrame({"x": [np.nan, 100.0], "c": ["b", "z"]})

    pre = preprocessor(numeric=("x",), categorical=("c",)).fit(train)
    with pytest.warns(UserWarning, match="unknown categories"):
        encoded = pre.transform(other)
    out = pd.DataFrame(encoded, columns=pre.get_feature_names_out())

    # Missing x is filled with the training median (2.0), not other's values.
    train_filled = np.array([1.0, 2.0, 2.0, 5.0])
    expected = (2.0 - train_filled.mean()) / train_filled.std()
    assert out.loc[0, "x"] == pytest.approx(expected)
    # "a" is the reference category; unseen "z" becomes all zeros.
    assert list(out.columns) == ["x", "c_b"]
    assert out["c_b"].tolist() == [1.0, 0.0]


def test_fit_logistic_learns_that_low_scores_default_more() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    train = pd.DataFrame(
        {
            "credit_score": rng.normal(740, 40, n),
            "ltv": rng.normal(75, 15, n),
            "dti": rng.normal(35, 8, n),
            "orig_rate": rng.normal(4, 0.5, n),
            "purpose": rng.choice(["C", "N", "P"], n),
            "occupancy": rng.choice(["I", "P", "S"], n),
        }
    )
    log_odds = -3 - 0.05 * (train["credit_score"] - 740)
    train["target"] = (rng.random(n) < 1 / (1 + np.exp(-log_odds))).astype(int)

    model = fit_logistic(train, load_config())
    coef = coefficients(model)

    assert coef.loc["credit_score", "coef"] < -1
    assert abs(coef.loc["dti", "coef"]) < 0.3
    assert len(coef) == 1 + 4 + 2 + 2


def test_fit_cox_recovers_hazard_ratio() -> None:
    loans = simulated_survival(20_000, seed=0)

    ratios = hazard_ratios(fit_cox(loans, load_config()))

    # One training std dev of score is about 40 points, so the ratio is near 0.5.
    assert ratios.loc["credit_score", "hazard_ratio"] == pytest.approx(0.5, abs=0.05)
    assert ratios.loc["dti", "hazard_ratio"] == pytest.approx(1.0, abs=0.1)


def test_default_by_rises_with_time_and_risk() -> None:
    loans = simulated_survival(5_000, seed=1)
    model = fit_cox(loans, load_config())
    safe = loans.assign(credit_score=800.0).head(1)
    risky = loans.assign(credit_score=650.0).head(1)

    assert default_by(model, safe, 12)[0] < default_by(model, safe, 24)[0]
    assert default_by(model, safe, 12)[0] < default_by(model, risky, 12)[0]
