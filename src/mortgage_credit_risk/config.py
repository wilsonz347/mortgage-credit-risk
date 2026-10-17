"""Load project settings from a YAML config file."""

from dataclasses import dataclass
from pathlib import Path

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
        Zero balance codes that mean prepayment.
    censor_codes : tuple of str
        Zero balance codes that end observation without an event.
    """

    min_delinquency_code: int
    reo_code: str
    credit_loss_codes: tuple[str, ...]
    ignore_relief_months: bool
    prepay_codes: tuple[str, ...]
    censor_codes: tuple[str, ...]


@dataclass(frozen=True)
class Config:
    """Project settings.

    Attributes
    ----------
    seed : int
        Random seed, also used to pick the loan sample.
    raw_dir : Path
        Folder with the Freddie Mac files and the ``header`` folder.
    interim_dir : Path
        Folder for caches.
    processed_dir : Path
        Folder for sampled, typed Parquet files.
    split : Split
        Vintage split and cutoff.
    events : Events
        Event codes.
    drop_harp : bool
        If true, HARP loans are left out.
    build_rate : float
        Share of loans to sample while building.
    final_rate : float
        Share of loans to sample for final results.
    """

    seed: int
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    split: Split
    events: Events
    drop_harp: bool
    build_rate: float
    final_rate: float


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
        interim_dir=(path.parent / data["interim_dir"]).resolve(),
        processed_dir=(path.parent / data["processed_dir"]).resolve(),
        split=Split(
            train_vintages=tuple(int(v) for v in split["train_vintages"]),
            validation_vintages=tuple(int(v) for v in split["validation_vintages"]),
            test_vintages=tuple(int(v) for v in split["test_vintages"]),
            cutoff_period=int(split["cutoff_period"]),
        ),
        events=Events(
            min_delinquency_code=int(events["default"]["min_delinquency_code"]),
            reo_code=str(events["default"]["reo_code"]),
            credit_loss_codes=tuple(
                str(c) for c in events["default"]["credit_loss_zero_balance_codes"]
            ),
            ignore_relief_months=bool(events["default"]["ignore_relief_months"]),
            prepay_codes=tuple(str(c) for c in events["prepay_zero_balance_codes"]),
            censor_codes=tuple(str(c) for c in events["censor_zero_balance_codes"]),
        ),
        drop_harp=bool(raw["filters"]["drop_harp"]),
        build_rate=float(raw["sample"]["build_rate"]),
        final_rate=float(raw["sample"]["final_rate"]),
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
