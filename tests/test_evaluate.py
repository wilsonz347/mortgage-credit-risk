"""Tests for evaluation helpers."""

import numpy as np
import pandas as pd
import pytest

from mortgage_credit_risk.evaluate import binary_metrics, calibration_table


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
