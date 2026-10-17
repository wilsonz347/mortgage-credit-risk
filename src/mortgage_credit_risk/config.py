"""Load project settings from a YAML config file."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


@dataclass(frozen=True)
class Split:
    """Vintage split and the calendar cutoff.

    Attributes
    ----------
    train_vintages : tuple of int
        Origination years used for training.
    validation_vintages : tuple of int
        Origination years used for validation.
    test_vintages : tuple of int
        Origination years used for test.
    cutoff_period : int
        First month of the test period, as YYYYMM. Training and validation
        only use months before it.
    """

    train_vintages: tuple[int, ...]
    validation_vintages: tuple[int, ...]
    test_vintages: tuple[int, ...]
    cutoff_period: int

    @property
    def vintages(self) -> tuple[int, ...]:
        """All origination years used in any split, in order."""
        return tuple(
            sorted(
                {*self.train_vintages, *self.validation_vintages, *self.test_vintages}
            )
        )


@dataclass(frozen=True)
class Events:
    """Codes that define default, prepayment and censoring.

    Attributes
    ----------
    min_delinquency_code : int
        Smallest delinquency status that counts as a default (3 = 90+ days).
    reo_code : str
        Delinquency status meaning the lender took the property.
    credit_loss_codes : tuple of str
        Zero balance codes that mean a credit loss.
    ignore_relief_months : bool
        If true, late months in forbearance or disaster relief do not count.
    prepay_codes : tuple of str
        Zero balance codes that mean prepayment. Any other ending is censored.
    """

    min_delinquency_code: int
    reo_code: str
    credit_loss_codes: tuple[str, ...]
    ignore_relief_months: bool
    prepay_codes: tuple[str, ...]


@dataclass(frozen=True)
class Macro:
    """Economic series from FRED and how they are joined to loans.

    Attributes
    ----------
    rate_series_long : str
        FRED ID of the market rate for long loans.
    rate_series_short : str
        FRED ID of the market rate for short loans.
    short_term_max_months : int
        Loans with a term this long or shorter use the short-loan rate.
    origination_lag_months : int
        Months between origination and the first payment.
    """

    rate_series_long: str
    rate_series_short: str
    short_term_max_months: int
    origination_lag_months: int


@dataclass(frozen=True)
class Config:
    """Project settings.

    Attributes
    ----------
    seed : int
        Random seed, also used to pick the loan sample.
    raw_dir : Path
        Folder with the Freddie Mac files and the ``header`` folder.
    fred_dir : Path
        Folder for cached FRED series.
    interim_dir : Path
        Folder for caches.
    processed_dir : Path
        Folder for sampled, typed Parquet files.
    split : Split
        Vintage split and cutoff.
    macro : Macro
        FRED series settings.
    events : Events
        Event codes.
    drop_harp : bool
        If true, HARP loans are left out.
    build_rate : float
        Share of loans to sample while building.
    final_rate : float
        Share of loans to sample for final results.
    logistic_horizon_months : int
        Loan age by which a default counts for the logistic baseline.
    numeric_features : tuple of str
        Numeric model inputs.
    categorical_features : tuple of str
        Categorical model inputs.
    logistic_params : dict
        Keyword arguments for ``LogisticRegression``.
    n_bins : int
        Number of risk bands in calibration tables.
    cox_params : dict
        Keyword arguments for ``CoxPHSurvivalAnalysis``.
    ph_split_age : int
        Loan age that splits the early and late windows in the proportional
        hazards check.
    n_bootstrap : int
        Number of resamples for bootstrap intervals.
    loss_horizon_months : int
        Loan age up to which expected loss is measured.
    """

    seed: int
    raw_dir: Path
    fred_dir: Path
    interim_dir: Path
    processed_dir: Path
    split: Split
    macro: Macro
    events: Events
    drop_harp: bool
    build_rate: float
    final_rate: float
    logistic_horizon_months: int
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    n_bins: int = 10
    ph_split_age: int = 18
    n_bootstrap: int = 1000
    loss_horizon_months: int = 36
    logistic_params: dict[str, Any] = field(default_factory=dict)
    cox_params: dict[str, Any] = field(default_factory=dict)


def load_config(path: Path = DEFAULT_CONFIG) -> Config:
    """Read a config file.

    Parameters
    ----------
    path : Path
        YAML config file. Relative paths inside it resolve against its folder.

    Returns
    -------
    Config
        Parsed settings.

    Raises
    ------
    ValueError
        If a vintage is in more than one split or a sample rate is not in (0, 1].
    """
    with path.open() as f:
        raw = yaml.safe_load(f)

    data, split, events = raw["data"], raw["split"], raw["events"]
    cfg = Config(
        seed=int(raw["seed"]),
        raw_dir=(path.parent / data["raw_dir"]).resolve(),
        fred_dir=(path.parent / data["fred_dir"]).resolve(),
        interim_dir=(path.parent / data["interim_dir"]).resolve(),
        processed_dir=(path.parent / data["processed_dir"]).resolve(),
        split=Split(
            train_vintages=tuple(int(v) for v in split["train_vintages"]),
            validation_vintages=tuple(int(v) for v in split["validation_vintages"]),
            test_vintages=tuple(int(v) for v in split["test_vintages"]),
            cutoff_period=int(split["cutoff_period"]),
        ),
        macro=Macro(
            rate_series_long=str(raw["macro"]["rate_series_long"]),
            rate_series_short=str(raw["macro"]["rate_series_short"]),
            short_term_max_months=int(raw["macro"]["short_term_max_months"]),
            origination_lag_months=int(raw["macro"]["origination_lag_months"]),
        ),
        events=Events(
            min_delinquency_code=int(events["default"]["min_delinquency_code"]),
            reo_code=str(events["default"]["reo_code"]),
            credit_loss_codes=tuple(
                str(c) for c in events["default"]["credit_loss_zero_balance_codes"]
            ),
            ignore_relief_months=bool(events["default"]["ignore_relief_months"]),
            prepay_codes=tuple(str(c) for c in events["prepay_zero_balance_codes"]),
        ),
        drop_harp=bool(raw["filters"]["drop_harp"]),
        build_rate=float(raw["sample"]["build_rate"]),
        final_rate=float(raw["sample"]["final_rate"]),
        logistic_horizon_months=int(raw["models"]["logistic_horizon_months"]),
        numeric_features=tuple(raw["models"]["numeric_features"]),
        categorical_features=tuple(raw["models"]["categorical_features"]),
        logistic_params=dict(raw["models"].get("logistic") or {}),
        n_bins=int(raw["evaluation"]["n_bins"]),
        ph_split_age=int(raw["evaluation"]["ph_split_age"]),
        n_bootstrap=int(raw["evaluation"]["n_bootstrap"]),
        loss_horizon_months=int(raw["loss"]["horizon_months"]),
        cox_params=dict(raw["models"].get("cox") or {}),
    )

    groups = [
        cfg.split.train_vintages,
        cfg.split.validation_vintages,
        cfg.split.test_vintages,
    ]
    if sum(len(g) for g in groups) != len(cfg.split.vintages):
        raise ValueError("A vintage is in more than one split")
    for rate in (cfg.build_rate, cfg.final_rate):
        if not 0 < rate <= 1:
            raise ValueError(f"Sample rate must be in (0, 1], got {rate}")
    return cfg
