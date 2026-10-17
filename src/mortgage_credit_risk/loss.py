"""Loss given default and expected loss."""

import pandas as pd

from mortgage_credit_risk.data import connect

CURRENT = "00"


def loan_outcomes(months_path: str, loan_ids: pd.Series, before: int) -> pd.DataFrame:
    """Summarize each loan's losses and status using months before a date.

    Parameters
    ----------
    months_path : str
        Loan-month Parquet file from ``build_sample`` (all months, with
        ``actual_loss``).
    loan_ids : pd.Series
        Loans to summarize.
    before : int
        Only months before this period (YYYYMM) are used.

    Returns
    -------
    pd.DataFrame
        Indexed by ``loan_id``: ``loss`` (dollars lost, positive),
        ``zero_balance_code`` (NaN if the loan was still open) and
        ``last_delinquency`` (status in the last month used).
    """
    con = connect()
    con.register("ids", pd.DataFrame({"loan_id": loan_ids}))
    out = con.execute(f"""
        SELECT m.loan_id,
            coalesce(sum(m.actual_loss), 0) AS loss,
            max(m.zero_balance_code) AS zero_balance_code,
            arg_max(m.delinquency, m.period) AS last_delinquency
        FROM '{months_path}' m JOIN ids USING (loan_id)
        WHERE m.period < {before}
        GROUP BY 1
    """).df()
    con.close()
    return out.set_index("loan_id")


def loss_given_default(defaults: pd.DataFrame, outcomes: pd.DataFrame) -> float:
    """Share of the original balance lost on defaulted loans with a known outcome.

    A default's outcome is known if the loan ended (any zero balance code) or
    was current again. Loans still late have an unknown outcome and are left
    out. Cured and paid-off loans count as no loss.

    Parameters
    ----------
    defaults : pd.DataFrame
        Defaulted loans with ``loan_id`` and ``orig_upb``.
    outcomes : pd.DataFrame
        Output of ``loan_outcomes`` for those loans.

    Returns
    -------
    float
        Dollars lost divided by original balance.
    """
    joined = defaults.set_index("loan_id").join(outcomes, how="inner")
    known = joined["zero_balance_code"].notna() | (
        joined["last_delinquency"] == CURRENT
    )
    resolved = joined[known]
    return float(resolved["loss"].sum() / resolved["orig_upb"].sum())


def expected_loss(pd_by_horizon: pd.Series, lgd: float, ead: pd.Series) -> pd.Series:
    """Compute expected loss per loan: default chance x loss rate x exposure.

    Parameters
    ----------
    pd_by_horizon : pd.Series
        Probability of default by the horizon.
    lgd : float
        Loss given default, as a share of exposure.
    ead : pd.Series
        Exposure at default in dollars.

    Returns
    -------
    pd.Series
        Expected loss in dollars.
    """
    return pd_by_horizon * lgd * ead
