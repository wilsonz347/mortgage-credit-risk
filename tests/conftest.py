"""Shared fixtures: a tiny, made-up copy of the raw Freddie Mac file layout.

The loans below are invented. Only the header lines match the real files.
"""

import re
from dataclasses import replace
from pathlib import Path

import pytest

from mortgage_credit_risk.config import Config, load_config

ORIG_HEADER = (
    "CLASSIC FICO|FIRST PAYMENT DATE|FIRST TIME HOMEBUYER INDICATOR|MATURITY DATE|"
    "METROPOLITAN STATISTICAL AREA (MSA) OR METROPOLITAN DIVISION|"
    "MORTGAGE INSURANCE PERCENTAGE (MI %)|NUMBER OF UNITS|OCCUPANCY STATUS|"
    "ORIGINAL COMBINED LOAN-TO-VALUE (CLTV)|ORIGINAL DEBT-TO-INCOME (DTI) RATIO|"
    "ORIGINAL UPB|ORIGINAL LOAN-TO-VALUE (LTV)|ORIGINAL INTEREST RATE|CHANNEL|"
    "PREPAYMENT PENALTY INDICATOR|AMORTIZATION TYPE|PROPERTY STATE|PROPERTY TYPE|"
    "POSTAL CODE|LOAN IDENTIFIER|LOAN PURPOSE|ORIGINAL LOAN TERM|NUMBER OF BORROWERS|"
    "SELLER NAME|SUPER CONFORMING FLAG|PRE-HARP LOAN SEQUENCE NUMBER|"
    "SPECIAL ELIGIBILITY PROGRAM|HARP INDICATOR|PROPERTY VALUATION METHOD|"
    "INTEREST ONLY (I/O) INDICATOR|VANTAGESCORE 4.0"
)
PERF_HEADER = (
    "LOAN IDENTIFIER|PERIOD|CURRENT ACTUAL UPB|CURRENT LOAN DELINQUENCY STATUS|"
    "LOAN AGE|REMAINING MONTHS TO LEGAL MATURITY|"
    "UNDERWRITING DEFECT AND MAJOR SERVICING DEFECT SETTLEMENT DATE|"
    "MODIFICATION FLAG|ZERO BALANCE CODE|ZERO BALANCE EFFECTIVE DATE|"
    "CURRENT INTEREST RATE|CURRENT NON-INTEREST BEARING UPB|"
    "DUE DATE OF LAST PAID INSTALLMENT (DDLPI)|MI RECOVERIES|NET SALES PROCEEDS|"
    "NON MI RECOVERIES|TOTAL EXPENSES|LEGAL COSTS|"
    "MAINTENANCE AND PRESERVATION COSTS|TAXES AND INSURANCE|MISCELLANEOUS EXPENSES|"
    "ACTUAL LOSS|CUMULATIVE MODIFICATION COSTS|INTEREST RATE STEP INDICATOR|"
    "PAYMENT DEFERRAL FLAG|ESTIMATED LOAN-TO-VALUE (ELTV)|ZERO BALANCE REMOVAL UPB|"
    "DELINQUENT ACCRUED INTEREST|DELINQUENCY DUE TO DISASTER|"
    "BORROWER ASSISTANCE PLAN|CURRENT PERIOD MODIFICATION COSTS|"
    "CURRENT INTEREST BEARING UPB|MORTGAGE INSURANCE CANCELLATION INDICATOR|"
    "SERVICER NAME|BANKRUPTCY CRAMDOWN COSTS"
)

ORIG_DEFAULTS = {
    "classic_fico": "760",
    "first_payment_date": "201503",
    "first_time_homebuyer_indicator": "N",
    "maturity_date": "204502",
    "metropolitan_statistical_area_msa_or_metropolitan_division": "14460",
    "mortgage_insurance_percentage_mi": "0",
    "number_of_units": "1",
    "occupancy_status": "P",
    "original_combined_loan_to_value_cltv": "80",
    "original_debt_to_income_dti_ratio": "35",
    "original_upb": "200000",
    "original_loan_to_value_ltv": "80",
    "original_interest_rate": "4.0",
    "channel": "R",
    "prepayment_penalty_indicator": "N",
    "amortization_type": "FRM",
    "property_state": "MA",
    "property_type": "SF",
    "postal_code": "021",
    "loan_purpose": "P",
    "original_loan_term": "360",
    "number_of_borrowers": "2",
    "seller_name": "OTHER",
    "super_conforming_flag": "N",
    "harp_indicator": "N",
    "property_valuation_method": "2",
    "interest_only_i_o_indicator": "N",
    "vantagescore_4_0": "9999",
}

# (quarter, loan id, changes to ORIG_DEFAULTS)
LOANS = [
    ("2015Q1", "F15Q10000001", {}),
    ("2015Q1", "F15Q10000002", {"harp_indicator": "Y"}),
    (
        "2015Q1",
        "F15Q10000003",
        {
            "classic_fico": "9999",
            "original_debt_to_income_dti_ratio": "999",
            "occupancy_status": "9",
        },
    ),
    ("2016Q2", "F16Q20000001", {}),
    ("2017Q3", "F17Q30000001", {}),
    ("2018Q1", "F18Q10000001", {}),
    ("2019Q4", "F19Q40000001", {}),
]

# (loan id, period, delinquency, zero balance code)
MONTHS = [
    ("F15Q10000001", "201502", "00", ""),
    ("F15Q10000001", "201503", "01", ""),
    ("F15Q10000001", "201504", "00", "01"),
    ("F15Q10000002", "201502", "00", ""),
    ("F15Q10000003", "201502", "00", ""),
    ("F15Q10000003", "201503", "RA", ""),
    ("F16Q20000001", "201605", "00", ""),
    ("F17Q30000001", "201708", "00", ""),
    ("F18Q10000001", "201802", "00", ""),
    ("F19Q40000001", "201911", "00", ""),
]


def snake(name: str) -> str:
    """Turn a header name into the snake_case name used by the package."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def write_lines(path: Path, lines: list[str]) -> None:
    """Write pipe-delimited lines, creating folders as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """Folder laid out like ``data/raw``, with a few made-up loans."""
    raw = tmp_path / "raw"
    (raw / "header").mkdir(parents=True)
    (raw / "header" / "origination_data_file_header.txt").write_text(ORIG_HEADER)
    (raw / "header" / "performance_data_file_header.txt").write_text(PERF_HEADER)

    orig_cols = [snake(c) for c in ORIG_HEADER.split("|")]
    perf_cols = [snake(c) for c in PERF_HEADER.split("|")]
    orig: dict[str, list[str]] = {}
    perf: dict[str, list[str]] = {}
    quarter_of = {}
    for quarter, loan_id, changes in LOANS:
        values = {**ORIG_DEFAULTS, **changes, "loan_identifier": loan_id}
        orig.setdefault(quarter, []).append(
            "|".join(values.get(c, "") for c in orig_cols)
        )
        quarter_of[loan_id] = quarter
    for loan_id, period, delinquency, zero_balance_code in MONTHS:
        values = {
            "loan_identifier": loan_id,
            "period": period,
            "current_actual_upb": "200000.00",
            "current_loan_delinquency_status": delinquency,
            "loan_age": "0",
            "zero_balance_code": zero_balance_code,
        }
        perf.setdefault(quarter_of[loan_id], []).append(
            "|".join(values.get(c, "") for c in perf_cols)
        )

    for quarter, lines in orig.items():
        folder = raw / f"historical_data_{quarter[:4]}" / f"historical_data_{quarter}"
        write_lines(folder / f"orig_{quarter}.txt", lines)
        write_lines(folder / f"perf_{quarter}.txt", perf[quarter])
    return raw


@pytest.fixture
def cfg(raw_dir: Path, tmp_path: Path) -> Config:
    """Default config pointed at the made-up raw files."""
    return replace(load_config(), raw_dir=raw_dir, processed_dir=tmp_path / "processed")
