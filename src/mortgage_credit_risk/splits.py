"""Assign loans to train, validation and test, and apply the calendar cutoff."""

from mortgage_credit_risk.config import Split

TRAIN = "train"
VALIDATION = "validation"
TEST = "test"


def int_list(values: tuple[int, ...]) -> str:
    """Format integers as a SQL list, e.g. ``(2015, 2016)``."""
    return "(" + ", ".join(str(v) for v in values) + ")"


def split_sql(source: str, split: Split) -> str:
    """Build a query that adds a ``split`` column and applies the cutoff.

    Loans are assigned by origination year. Train and validation rows from
    the cutoff month on are dropped, so any event after the cutoff is not
    seen and the loan counts as censored. Test rows are all kept. Loans from
    years in no split are dropped.

    Run this after labeling events, so that an event after the cutoff is
    removed together with its row.

    Parameters
    ----------
    source : str
        SQL table, view or subquery with ``vintage`` and ``period`` columns.
    split : Split
        Vintage split and cutoff from the config.

    Returns
    -------
    str
        SQL query with every column of ``source`` plus ``split``.
    """
    train = int_list(split.train_vintages)
    validation = int_list(split.validation_vintages)
    test = int_list(split.test_vintages)
    return f"""
        SELECT *,
            CASE
                WHEN vintage IN {train} THEN '{TRAIN}'
                WHEN vintage IN {validation} THEN '{VALIDATION}'
                ELSE '{TEST}'
            END AS split
        FROM {source}
        WHERE (vintage IN {train} AND period < {split.cutoff_period})
            OR (vintage IN {validation} AND period < {split.cutoff_period})
            OR vintage IN {test}
    """
