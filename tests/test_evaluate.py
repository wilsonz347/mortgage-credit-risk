"""Tests for evaluation helpers."""

import numpy as np
import pandas as pd
import pytest

from mortgage_credit_risk.config import load_config
from mortgage_credit_risk.evaluate import (
    binary_metrics,
    c_index,
    calibration_table,
    hazard_ratios_by_window,
    paired_bootstrap_auc,
)
from mortgage_credit_risk.models import preprocessor
from tests.simulate import simulated_survival


def test_binary_metrics_on_calibrated_predictions() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.2, 50_000)
    y = pd.Series((rng.random(p.size) < p).astype(int))

    m = binary_metrics(y, p)

    assert m["auc"] > 0.6
    assert m["ratio"] == pytest.approx(1.0, abs=0.03)


def test_binary_metrics_random_ranking_has_auc_near_half() -> None:
    rng = np.random.default_rng(1)
    y = pd.Series(rng.integers(0, 2, 20_000))

    assert binary_metrics(y, rng.random(y.size))["auc"] == pytest.approx(0.5, abs=0.02)


def test_calibration_table_bands_are_equal_and_ordered() -> None:
    p = np.linspace(0.0, 0.1, 1000)
    y = pd.Series(np.r_[np.zeros(900), np.ones(100)].astype(int))

    table = calibration_table(y, p, n_bins=10)

    assert table["loans"].tolist() == [100] * 10
    assert table["mean_predicted"].is_monotonic_increasing
    assert table.loc[10, "events"] == 100


def test_c_index_perfect_and_reversed() -> None:
    loans = pd.DataFrame({"last_age": [3, 6, 9, 12], "event": [1, 1, 1, 0]})
    earlier_is_riskier = np.array([4.0, 3.0, 2.0, 1.0])

    assert c_index(loans, earlier_is_riskier) == 1.0
    assert c_index(loans, -earlier_is_riskier) == 0.0


def test_hazard_ratios_by_window_agree_when_hazards_are_proportional() -> None:
    loans = simulated_survival(30_000, seed=2)
    cfg = load_config()
    pre = preprocessor(cfg.numeric_features, cfg.categorical_features).fit(loans)

    table = hazard_ratios_by_window(loans, pre, split_age=12)
    score = table.loc["credit_score"].set_index("window")["hazard_ratio"]

    assert score["early"] == pytest.approx(0.5, abs=0.07)
    assert score["late"] == pytest.approx(0.5, abs=0.07)


def test_paired_bootstrap_prefers_the_informative_model() -> None:
    rng = np.random.default_rng(3)
    p = rng.uniform(0.01, 0.3, 5000)
    y = pd.Series((rng.random(p.size) < p).astype(int))
    noise = rng.random(p.size)

    result = paired_bootstrap_auc(y, p, noise, n=200, seed=0)

    assert result["low_95"] > 0
    assert result["share_a_wins"] == 1.0
