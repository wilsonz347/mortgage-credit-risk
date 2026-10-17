"""Read the Freddie Mac files, sample loans and save typed Parquet tables."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import duckdb

from mortgage_credit_risk.config import Config

logger = logging.getLogger(__name__)

HEADER_FILES = {
    "orig": "origination_data_file_header.txt",
    "perf": "performance_data_file_header.txt",
}


@dataclass(frozen=True)
class Column:
    """How to read one raw column.

    Attributes
    ----------
    raw : str
        Snake_case name from the header file.
    name : str
        Name in the output table.
    dtype : str
        DuckDB type to cast to.
    missing : str or None
        Code that means "not available". It becomes NULL.
    """

    raw: str
    name: str
    dtype: str = "VARCHAR"
    missing: str | None = None


# Codes for "not available" are from the Freddie Mac General User Guide.
# VantageScore (always missing), amortization type, interest-only and
# prepayment penalty (one value each) and the pre-HARP loan number are left out.
ORIG_COLUMNS = [
    Column("loan_identifier", "loan_id"),
    Column("classic_fico", "credit_score", "INTEGER", "9999"),
    Column("first_payment_date", "first_payment_period", "INTEGER"),
    Column("first_time_homebuyer_indicator", "first_time_buyer", missing="9"),
    Column("maturity_date", "maturity_period", "INTEGER"),
    Column("metropolitan_statistical_area_msa_or_metropolitan_division", "msa"),
    Column("mortgage_insurance_percentage_mi", "mi_pct", "INTEGER", "999"),
    Column("number_of_units", "units", "INTEGER", "99"),
    Column("occupancy_status", "occupancy", missing="9"),
    Column("original_combined_loan_to_value_cltv", "cltv", "INTEGER", "999"),
    Column("original_debt_to_income_dti_ratio", "dti", "INTEGER", "999"),
    Column("original_upb", "orig_upb", "DOUBLE"),
    Column("original_loan_to_value_ltv", "ltv", "INTEGER", "999"),
    Column("original_interest_rate", "orig_rate", "DOUBLE"),
    Column("channel", "channel", missing="9"),
    Column("property_state", "state"),
    Column("property_type", "property_type", missing="99"),
    Column("postal_code", "postal_code"),
    Column("loan_purpose", "purpose", missing="9"),
    Column("original_loan_term", "orig_term", "INTEGER"),
    Column("number_of_borrowers", "n_borrowers", "INTEGER", "99"),
    Column("seller_name", "seller"),
    Column("super_conforming_flag", "super_conforming"),
    Column("special_eligibility_program", "program", missing="9"),
    Column("harp_indicator", "harp"),
    Column("property_valuation_method", "valuation_method", "INTEGER", "7"),
]

# Delinquency status stays text because it includes "RA".
PERF_COLUMNS = [
    Column("loan_identifier", "loan_id"),
    Column("period", "period", "INTEGER"),
    Column("loan_age", "loan_age", "INTEGER"),
    Column("remaining_months_to_legal_maturity", "months_to_maturity", "INTEGER"),
    Column("current_actual_upb", "current_upb", "DOUBLE"),
    Column("current_interest_rate", "current_rate", "DOUBLE"),
    Column("current_loan_delinquency_status", "delinquency"),
    Column("modification_flag", "modification_flag"),
    Column("payment_deferral_flag", "payment_deferral_flag"),
    Column("borrower_assistance_plan", "assistance_plan"),
    Column("delinquency_due_to_disaster", "disaster_flag"),
    Column("zero_balance_code", "zero_balance_code"),
    Column("zero_balance_effective_date", "zero_balance_period", "INTEGER"),
    Column("zero_balance_removal_upb", "removal_upb", "DOUBLE"),
    Column("actual_loss", "actual_loss", "DOUBLE"),
]


def read_header(raw_dir: Path, kind: str) -> list[str]:
    """Read a header file and return snake_case column names.

    Parameters
    ----------
    raw_dir : Path
        Folder that contains the ``header`` folder.
    kind : {"orig", "perf"}
        Which file type.

    Returns
    -------
    list of str
        Column names in file order.
    """
    text = (raw_dir / "header" / HEADER_FILES[kind]).read_text().strip()
    return [re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_") for c in text.split("|")]


def raw_files(raw_dir: Path, kind: str, vintages: tuple[int, ...]) -> list[str]:
    """List the raw files of one type for the given origination years.

    Parameters
    ----------
    raw_dir : Path
        Folder with one ``historical_data_YYYY`` folder per year.
    kind : {"orig", "perf"}
        Which file type.
    vintages : tuple of int
        Origination years.

    Returns
    -------
    list of str
        Sorted file paths.

    Raises
    ------
    FileNotFoundError
        If a year has no files.
    """
    files = []
    for year in vintages:
        found = sorted(raw_dir.glob(f"historical_data_{year}/*/{kind}_{year}Q*.txt"))
        if not found:
            raise FileNotFoundError(f"No {kind} files for {year} under {raw_dir}")
        files += [str(f) for f in found]
    return files


def select_sql(raw_dir: Path, kind: str, vintages: tuple[int, ...]) -> str:
    """Build a query that reads raw files and returns typed, renamed columns.

    Every raw column is read as text first, so "not available" codes can be
    turned into NULL before casting.

    Parameters
    ----------
    raw_dir : Path
        Folder with the raw files and the ``header`` folder.
    kind : {"orig", "perf"}
        Which file type.
    vintages : tuple of int
        Origination years.

    Returns
    -------
    str
        SQL query. Origination rows also get ``vintage`` and ``orig_quarter``.
    """
    columns = ORIG_COLUMNS if kind == "orig" else PERF_COLUMNS
    parts = []
    for col in columns:
        value = col.raw
        if col.missing is not None:
            value = f"NULLIF({value}, '{col.missing}')"
        parts.append(f"CAST({value} AS {col.dtype}) AS {col.name}")
    if kind == "orig":
        quarter = r"regexp_extract(filename, '(\d{4}Q\d)', 1)"
        parts += [
            f"CAST(left({quarter}, 4) AS INTEGER) AS vintage",
            f"{quarter} AS orig_quarter",
        ]
    return f"""
        SELECT {", ".join(parts)}
        FROM read_csv(
            {raw_files(raw_dir, kind, vintages)},
            delim = '|',
            header = false,
            names = {read_header(raw_dir, kind)},
            all_varchar = true,
            filename = true
        )
    """


def in_sample_sql(id_column: str, seed: int, rate: float) -> str:
    """Build a condition that keeps a fixed share of loans.

    Each loan gets a number in [0, 1) from the MD5 hash of the seed and its
    ID. A loan is kept if that number is below ``rate``. The same seed always
    keeps the same loans, and a smaller sample is always part of a larger one.

    Parameters
    ----------
    id_column : str
        Column with the loan ID.
    seed : int
        Seed mixed into the hash.
    rate : float
        Share of loans to keep, in (0, 1].

    Returns
    -------
    str
        SQL condition.
    """
    hashed = f"CAST(('0x' || left(md5('{seed}' || {id_column}), 8)) AS UBIGINT)"
    return f"{hashed} / 4294967296.0 < {rate}"


def build_sample(cfg: Config, rate: float) -> tuple[Path, Path]:
    """Sample loans and save their origination and monthly rows as Parquet.

    Reads only the vintages in the split. Drops HARP loans if the config says
    so. Reading the performance files takes several minutes.

    Parameters
    ----------
    cfg : Config
        Project settings.
    rate : float
        Share of loans to keep, in (0, 1].

    Returns
    -------
    tuple of Path
        Paths of the loans table (one row per loan) and the loan-months
        table (one row per loan per month).
    """
    vintages = cfg.split.vintages
    tag = f"{rate * 100:g}pct"
    loans_path = cfg.processed_dir / f"loans_{tag}.parquet"
    months_path = cfg.processed_dir / f"loan_months_{tag}.parquet"
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)

    keep = in_sample_sql("loan_id", cfg.seed, rate)
    if cfg.drop_harp:
        keep += " AND coalesce(harp, 'N') <> 'Y'"

    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    logger.info("Writing %s", loans_path)
    con.execute(f"""
        COPY (
            SELECT * FROM ({select_sql(cfg.raw_dir, "orig", vintages)})
            WHERE {keep}
            ORDER BY loan_id
        ) TO '{loans_path}' (FORMAT parquet)
    """)

    logger.info("Writing %s", months_path)
    con.execute(f"""
        COPY (
            SELECT * FROM ({select_sql(cfg.raw_dir, "perf", vintages)})
            WHERE {in_sample_sql("loan_id", cfg.seed, rate)}
                AND loan_id IN (SELECT loan_id FROM '{loans_path}')
            ORDER BY loan_id, period
        ) TO '{months_path}' (FORMAT parquet)
    """)
    con.close()
    return loans_path, months_path
