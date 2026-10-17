"""Tests for config loading."""

from pathlib import Path

import pytest

from mortgage_credit_risk.config import DEFAULT_CONFIG, load_config


def test_default_config_loads() -> None:
    cfg = load_config()

    assert cfg.raw_dir == DEFAULT_CONFIG.parents[1] / "data" / "raw"
    assert cfg.processed_dir.is_absolute()
    assert cfg.split.vintages == (2015, 2016, 2017, 2019)
    assert cfg.split.cutoff_period == 201901
    assert cfg.events.min_delinquency_code == 3
    assert "01" in cfg.events.prepay_codes
    assert 0 < cfg.build_rate <= cfg.final_rate <= 1
    assert cfg.logistic_horizon_months == 12
    assert "credit_score" in cfg.numeric_features
    assert cfg.logistic_params["C"] == 1.0
    assert cfg.n_bins == 10


def write_config(tmp_path: Path, validation: str = "[2017]", rate: str = "0.1") -> Path:
    path = tmp_path / "configs" / "test.yaml"
    path.parent.mkdir()
    path.write_text(
        "seed: 1\n"
        "data: {raw_dir: ../data/raw, interim_dir: ../data/i, processed_dir: ../p}\n"
        "split:\n"
        "  train_vintages: [2015, 2016]\n"
        f"  validation_vintages: {validation}\n"
        "  test_vintages: [2019]\n"
        "  cutoff_period: 201901\n"
        "events:\n"
        "  default:\n"
        "    min_delinquency_code: 3\n"
        "    reo_code: RA\n"
        "    credit_loss_zero_balance_codes: ['02']\n"
        "    ignore_relief_months: true\n"
        "  prepay_zero_balance_codes: ['01']\n"
        "  censor_zero_balance_codes: ['15']\n"
        "filters: {drop_harp: true}\n"
        f"sample: {{build_rate: {rate}, final_rate: 0.2}}\n"
        "models:\n"
        "  logistic_horizon_months: 12\n"
        "  numeric_features: [ltv]\n"
        "  categorical_features: [purpose]\n"
        "evaluation: {n_bins: 5}\n"
    )
    return path


def test_relative_paths_resolve_against_config_folder(tmp_path: Path) -> None:
    cfg = load_config(write_config(tmp_path))

    assert cfg.raw_dir == (tmp_path / "data" / "raw").resolve()
    assert cfg.processed_dir == (tmp_path / "p").resolve()
    assert cfg.events.credit_loss_codes == ("02",)


def test_vintage_in_two_splits_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="more than one split"):
        load_config(write_config(tmp_path, validation="[2016]"))


def test_bad_sample_rate_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Sample rate"):
        load_config(write_config(tmp_path, rate="1.5"))
