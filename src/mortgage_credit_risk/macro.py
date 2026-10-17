"""Download economic series from FRED and turn them into monthly values."""

import json
import logging
import os
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from mortgage_credit_risk.config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
ENV_FILE = DEFAULT_CONFIG.parents[1] / ".env"
KEY_NAME = "FRED_API_KEY"


def fred_api_key(env_file: Path = ENV_FILE) -> str:
    """Read the FRED API key from the environment or a ``.env`` file.

    Parameters
    ----------
    env_file : Path
        File with a ``FRED_API_KEY=...`` line. Used if the environment
        variable is not set.

    Returns
    -------
    str
        The key.

    Raises
    ------
    RuntimeError
        If no key is found.
    """
    key = os.environ.get(KEY_NAME)
    if not key and env_file.exists():
        for line in env_file.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == KEY_NAME:
                key = value.strip().strip("'\"")
    if not key:
        raise RuntimeError(f"Set {KEY_NAME} in the environment or in {env_file}")
    return key


def fetch_series(series_id: str, api_key: str) -> pd.Series:
    """Download all observations of one FRED series.

    Parameters
    ----------
    series_id : str
        FRED series ID, e.g. ``MORTGAGE30US``.
    api_key : str
        FRED API key.

    Returns
    -------
    pd.Series
        Values indexed by date. FRED marks missing values with ``.``; they
        become NaN.
    """
    query = urllib.parse.urlencode(
        {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    )
    with urllib.request.urlopen(f"{FRED_URL}?{query}", timeout=60) as response:
        observations = json.load(response)["observations"]
    dates = pd.to_datetime([o["date"] for o in observations])
    values = pd.to_numeric([o["value"] for o in observations], errors="coerce")
    return pd.Series(values, index=pd.DatetimeIndex(dates, name="date"), name=series_id)


def load_series(series_id: str, fred_dir: Path) -> pd.Series:
    """Load a FRED series from the cache, downloading it the first time.

    Parameters
    ----------
    series_id : str
        FRED series ID.
    fred_dir : Path
        Cache folder. The series is saved as ``<series_id>.csv``.

    Returns
    -------
    pd.Series
        Values indexed by date.
    """
    path = fred_dir / f"{series_id}.csv"
    if not path.exists():
        logger.info("Downloading %s from FRED", series_id)
        fred_dir.mkdir(parents=True, exist_ok=True)
        fetch_series(series_id, fred_api_key()).to_csv(path)
    series = pd.read_csv(path, index_col="date", parse_dates=["date"]).iloc[:, 0]
    return series.rename(series_id)


def monthly_mean(series: pd.Series) -> pd.Series:
    """Average a series within each calendar month.

    Parameters
    ----------
    series : pd.Series
        Values indexed by date (weekly or daily).

    Returns
    -------
    pd.Series
        Monthly averages indexed by period as an integer YYYYMM.
    """
    clean = series.dropna()
    dates = pd.DatetimeIndex(clean.index)
    period = dates.year * 100 + dates.month
    monthly = clean.groupby(period).mean()
    monthly.index.name = "period"
    return monthly
