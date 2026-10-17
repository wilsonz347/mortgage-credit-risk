"""Build the modeling tables from the raw files.

Run with ``python -m mortgage_credit_risk.pipeline``.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from mortgage_credit_risk.config import DEFAULT_CONFIG, Config, load_config
from mortgage_credit_risk.data import build_sample, connect, sample_paths, sample_tag
from mortgage_credit_risk.events import label_sql
from mortgage_credit_risk.macro import load_series, monthly_mean
from mortgage_credit_risk.splits import TEST, split_sql

logger = logging.getLogger(__name__)

# Origination fields known on day one, carried onto every loan-month row.
FEATURES = [
    "credit_score",
    "ltv",
    "cltv",
    "dti",
    "orig_rate",
    "orig_upb",
    "orig_term",
    "mi_pct",
    "units",
    "n_borrowers",
    "first_time_buyer",
    "occupancy",
    "purpose",
    "channel",
    "property_type",
    "state",
    "super_conforming",
    "program",
    "market_rate",
    "rate_spread",
]


def market_rates(cfg: Config) -> pd.DataFrame:
    """Monthly average market mortgage rates for long and short loans.

    Parameters
    ----------
    cfg : Config
        Project settings with the FRED series IDs and cache folder.

    Returns
    -------
    pd.DataFrame
        Columns ``period`` (YYYYMM), ``long_rate`` and ``short_rate``.
    """
    long_rate = monthly_mean(load_series(cfg.macro.rate_series_long, cfg.fred_dir))
    short_rate = monthly_mean(load_series(cfg.macro.rate_series_short, cfg.fred_dir))
    rates = pd.concat({"long_rate": long_rate, "short_rate": short_rate}, axis=1)
    return rates.reset_index()


def build_model_tables(cfg: Config, rate: float) -> tuple[Path, Path]:
    """Write the loan-month table and the loan table used for modeling.

    The loan-month table has one row per loan per month, up to the loan's
    first event, the cutoff (train and validation) or the end of the data.
    ``event`` is 0 (none), 1 (default) or 2 (prepay) for that month. This is
    the input for discrete-time hazard models.

    The loan table has one row per loan: ``months`` observed, ``last_age``
    (loan age in the last observed month, which is the event month if there
    is one), the final ``event`` (0 means censored) and ``event_period``.
    This is the input for Cox models and fixed-horizon baselines.

    Both tables carry ``split``, ``vintage`` and the origination features in
    ``FEATURES``. The sample is built first if it does not exist yet.

    Parameters
    ----------
    cfg : Config
        Project settings.
    rate : float
        Share of loans to use, in (0, 1].

    Returns
    -------
    tuple of Path
        Paths of the loan-month table and the loan table.

    Raises
    ------
    ValueError
        If a test row falls before the cutoff.
    """
    tag = sample_tag(rate)
    loans_path, months_path = sample_paths(cfg, rate)
    if not (loans_path.exists() and months_path.exists()):
        loans_path, months_path = build_sample(cfg, rate)

    hazard_path = cfg.processed_dir / f"model_loan_months_{tag}.parquet"
    loan_table_path = cfg.processed_dir / f"model_loans_{tag}.parquet"
    features = ", ".join(f"l.{f}" for f in FEATURES)

    con = connect()
    con.register("rates", market_rates(cfg))
    lag = cfg.macro.origination_lag_months
    con.execute(f"""
        CREATE VIEW loans AS
        WITH dated AS (
            SELECT *,
                CAST(strftime(
                    strptime(CAST(first_payment_period AS VARCHAR), '%Y%m')
                        - INTERVAL {lag} MONTH,
                    '%Y%m'
                ) AS INTEGER) AS orig_period
            FROM '{loans_path}'
        )
        SELECT d.*,
            CASE WHEN d.orig_term <= {cfg.macro.short_term_max_months}
                THEN r.short_rate ELSE r.long_rate END AS market_rate,
            d.orig_rate - market_rate AS rate_spread
        FROM dated d LEFT JOIN rates r ON r.period = d.orig_period
    """)
    con.execute(f"""
        CREATE VIEW months AS
        SELECT m.*, l.vintage
        FROM '{months_path}' m JOIN loans l USING (loan_id)
    """)
    con.execute(f"""
        CREATE VIEW labeled AS
        {split_sql(f"({label_sql('months', cfg.events)})", cfg.split)}
    """)

    early = con.execute(f"""
        SELECT count(*) FROM labeled
        WHERE split = '{TEST}' AND period < {cfg.split.cutoff_period}
    """).fetchone()
    if early is not None and early[0] > 0:
        raise ValueError(f"{early[0]} test rows fall before the cutoff")

    logger.info("Writing %s", hazard_path)
    con.execute(f"""
        COPY (
            SELECT m.split, m.loan_id, m.vintage, m.period, m.loan_age, m.event,
                {features}
            FROM labeled m JOIN loans l USING (loan_id)
            ORDER BY m.loan_id, m.period
        ) TO '{hazard_path}' (FORMAT parquet)
    """)

    logger.info("Writing %s", loan_table_path)
    con.execute(f"""
        COPY (
            SELECT m.split, m.loan_id, m.vintage, m.months, m.last_age,
                m.event, m.event_period,
                {features}
            FROM (
                SELECT loan_id,
                    any_value(split) AS split,
                    any_value(vintage) AS vintage,
                    count(*) AS months,
                    max(loan_age) AS last_age,
                    max(event) AS event,
                    max(period) FILTER (WHERE event > 0) AS event_period
                FROM labeled GROUP BY loan_id
            ) m JOIN loans l USING (loan_id)
            ORDER BY m.loan_id
        ) TO '{loan_table_path}' (FORMAT parquet)
    """)
    con.close()
    return hazard_path, loan_table_path


def main(argv: list[str] | None = None) -> None:
    """Build the modeling tables from the command line.

    Parameters
    ----------
    argv : list of str, optional
        Arguments. Defaults to ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = load_config(args.config)
    for path in build_model_tables(cfg, cfg.sample_rate):
        print(path)


if __name__ == "__main__":
    main()
