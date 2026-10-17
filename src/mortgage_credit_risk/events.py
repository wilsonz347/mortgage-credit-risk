"""Label each loan-month with the event that happens in it."""

from mortgage_credit_risk.config import Events

NO_EVENT = 0
DEFAULT = 1
PREPAY = 2

# Freddie Mac codes for payment relief. See the General User Guide.
FORBEARANCE = "F"
DISASTER = "Y"


def sql_list(codes: tuple[str, ...]) -> str:
    """Format codes as a SQL list, e.g. ``('02', '03')``."""
    return "(" + ", ".join(f"'{c}'" for c in codes) + ")"


def label_sql(source: str, events: Events) -> str:
    """Build a query that adds an ``event`` column and drops rows after it.

    A month is a default if the loan is 90+ days late (or the lender took the
    property) or it ends with a credit loss. If ``events.ignore_relief_months``
    is true, late months in forbearance or disaster relief do not count. A
    month is a prepayment if the loan ends with a prepayment code. Default
    wins if both happen in the same month. Each loan keeps its rows up to and
    including its first event. Loans that end with a censoring code, or never
    end, keep all rows with no event.

    Parameters
    ----------
    source : str
        SQL table, view or subquery with one row per loan per month and the
        columns ``loan_id``, ``period``, ``delinquency``, ``assistance_plan``,
        ``disaster_flag`` and ``zero_balance_code``.
    events : Events
        Event codes from the config.

    Returns
    -------
    str
        SQL query with every column of ``source`` plus ``event``
        (0 none, 1 default, 2 prepay).
    """
    late = (
        f"(TRY_CAST(delinquency AS INTEGER) >= {events.min_delinquency_code}"
        f" OR delinquency = '{events.reo_code}')"
    )
    if events.ignore_relief_months:
        relief = (
            f"coalesce(assistance_plan = '{FORBEARANCE}'"
            f" OR disaster_flag = '{DISASTER}', false)"
        )
        late = f"({late} AND NOT {relief})"
    credit_loss = f"zero_balance_code IN {sql_list(events.credit_loss_codes)}"
    prepay = f"zero_balance_code IN {sql_list(events.prepay_codes)}"

    return f"""
        WITH flagged AS (
            SELECT *,
                CASE
                    WHEN {late} OR {credit_loss} THEN {DEFAULT}
                    WHEN {prepay} THEN {PREPAY}
                    ELSE {NO_EVENT}
                END AS event
            FROM {source}
        ),
        first_event AS (
            SELECT loan_id, min(period) FILTER (WHERE event <> {NO_EVENT}) AS period
            FROM flagged
            GROUP BY loan_id
        )
        SELECT flagged.*
        FROM flagged JOIN first_event USING (loan_id)
        WHERE first_event.period IS NULL OR flagged.period <= first_event.period
    """
