"""Ranking and calibration checks for predicted default probabilities."""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def binary_metrics(y: pd.Series, p: np.ndarray) -> dict[str, float]:
    """Summarize how well predicted probabilities match 0/1 outcomes.

    Parameters
    ----------
    y : pd.Series
        Observed outcomes, 0 or 1.
    p : np.ndarray
        Predicted probabilities of a 1.

    Returns
    -------
    dict of str to float
        ``auc`` (ranking, 0.5 is random), ``log_loss`` and ``brier``
        (lower is better), ``mean_predicted`` and ``observed`` (rates), and
        ``ratio`` (mean predicted / observed; 1 is calibrated overall).
    """
    observed = float(np.mean(y))
    mean_predicted = float(np.mean(p))
    return {
        "auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "mean_predicted": mean_predicted,
        "observed": observed,
        "ratio": mean_predicted / observed,
    }


def calibration_table(y: pd.Series, p: np.ndarray, n_bins: int) -> pd.DataFrame:
    """Compare predicted and observed rates within bands of predicted risk.

    Loans are sorted by prediction and cut into ``n_bins`` groups of equal
    size. Band 1 is the lowest predicted risk.

    Parameters
    ----------
    y : pd.Series
        Observed outcomes, 0 or 1.
    p : np.ndarray
        Predicted probabilities of a 1.
    n_bins : int
        Number of bands.

    Returns
    -------
    pd.DataFrame
        One row per band: ``loans``, ``events``, ``mean_predicted`` and
        ``observed`` rates.
    """
    df = pd.DataFrame({"y": np.asarray(y), "p": p})
    df["band"] = pd.qcut(df["p"].rank(method="first"), n_bins, labels=False) + 1
    return df.groupby("band").agg(
        loans=("y", "size"),
        events=("y", "sum"),
        mean_predicted=("p", "mean"),
        observed=("y", "mean"),
    )
