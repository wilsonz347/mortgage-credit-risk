"""Tests for FRED downloads and monthly conversion."""

import io
import json
from pathlib import Path

import pandas as pd
import pytest

from mortgage_credit_risk import macro


def test_api_key_from_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("OTHER=1\nFRED_API_KEY='abc123'\n")

    assert macro.fred_api_key(env) == "abc123"


def test_api_key_environment_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FRED_API_KEY", "from_env")
    (tmp_path / ".env").write_text("FRED_API_KEY=from_file\n")

    assert macro.fred_api_key(tmp_path / ".env") == "from_env"


def test_missing_api_key_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="FRED_API_KEY"):
        macro.fred_api_key(tmp_path / "missing.env")


def test_fetch_series_parses_values_and_missing_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = {
        "observations": [
            {"date": "2016-03-03", "value": "3.64"},
            {"date": "2016-03-10", "value": "."},
        ]
    }
    monkeypatch.setattr(
        macro.urllib.request,
        "urlopen",
        lambda url, timeout: io.BytesIO(json.dumps(body).encode()),
    )

    series = macro.fetch_series("MORTGAGE30US", "key")

    assert series.iloc[0] == 3.64
    assert pd.isna(series.iloc[1])
    assert series.name == "MORTGAGE30US"


def test_load_series_downloads_once_then_uses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_fetch(series_id: str, api_key: str) -> pd.Series:
        calls.append(series_id)
        index = pd.DatetimeIndex(["2016-03-03"], name="date")
        return pd.Series([3.64], index=index, name=series_id)

    monkeypatch.setattr(macro, "fetch_series", fake_fetch)
    monkeypatch.setenv("FRED_API_KEY", "key")

    first = macro.load_series("X", tmp_path)
    second = macro.load_series("X", tmp_path)

    assert calls == ["X"]
    assert first.equals(second)


def test_monthly_mean_averages_weeks_and_skips_missing() -> None:
    weeks = pd.to_datetime(["2016-03-03", "2016-03-10", "2016-03-17", "2016-04-07"])
    series = pd.Series([3.6, 3.8, float("nan"), 4.0], index=weeks)

    assert macro.monthly_mean(series).to_dict() == {201603: 3.7, 201604: 4.0}
