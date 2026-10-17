"""Ranking and calibration checks for predicted default probabilities."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sksurv.metrics import concordance_index_censored
from statsmodels.duration.hazard_regression import PHReg

from mortgage_credit_risk.events import DEFAULT


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


def c_index(loans: pd.DataFrame, risk: np.ndarray) -> float:
    """Harrell's concordance index for default, allowing for censoring.

    Among pairs of loans where we know which defaulted first, the share in
    which that loan has the higher risk score.

    Parameters
    ----------
    loans : pd.DataFrame
        Loan table with ``last_age`` and ``event``.
    risk : np.ndarray
        Risk score per loan, higher meaning riskier.

    Returns
    -------
    float
        C-index: 0.5 is random, 1 is perfect.
    """
    result = concordance_index_censored(
        (loans["event"] == DEFAULT).to_numpy(), loans["last_age"].to_numpy(), risk
    )
    return float(result[0])


def hazard_ratios_by_window(
    loans: pd.DataFrame, pre: ColumnTransformer, split_age: int
) -> pd.DataFrame:
    """Fit Cox separately on young and older loans to check proportional hazards.

    The early window follows every loan until ``split_age``. The late window
    follows only loans still active at ``split_age``, from that age on (left
    truncation). Under proportional hazards, each hazard ratio is about the
    same in both windows.

    Parameters
    ----------
    loans : pd.DataFrame
        Loan table with ``last_age``, ``event`` and the features.
    pre : ColumnTransformer
        Preprocessing already fit on training data.
    split_age : int
        Loan age where the late window starts.

    Returns
    -------
    pd.DataFrame
        Hazard ratio with a 95% interval for each input and window, plus the
        number of defaults in each window.
    """
    x = pre.transform(loans)
    age = loans["last_age"].to_numpy()
    default = (loans["event"] == DEFAULT).to_numpy()
    names = pre.get_feature_names_out()
    late = age > split_age

    early = window_hazard_ratios(
        x, np.minimum(age, split_age), default & (age <= split_age), None, names
    )
    later = window_hazard_ratios(
        x[late], age[late], default[late], np.full(late.sum(), split_age), names
    )
    return pd.concat([early.assign(window="early"), later.assign(window="late")])


def window_hazard_ratios(
    x: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    entry: np.ndarray | None,
    names: np.ndarray,
) -> pd.DataFrame:
    """Fit one Cox model with statsmodels and return its hazard ratios.

    Parameters
    ----------
    x : np.ndarray
        Preprocessed inputs.
    time : np.ndarray
        Age when each loan leaves the window.
    event : np.ndarray
        True if the loan defaults at ``time``.
    entry : np.ndarray or None
        Age when each loan enters the window. None means age 0.
    names : np.ndarray
        Input names.

    Returns
    -------
    pd.DataFrame
        Hazard ratio, 95% interval and number of defaults, one row per input.
    """
    fit = PHReg(time, x, status=event.astype(int), entry=entry, ties="breslow").fit()
    interval = np.exp(fit.conf_int())
    return pd.DataFrame(
        {
            "hazard_ratio": np.exp(fit.params),
            "low_95": interval[:, 0],
            "high_95": interval[:, 1],
            "defaults": int(event.sum()),
        },
        index=pd.Index(names, name="feature"),
    )


def paired_bootstrap_auc(
    y: pd.Series, p_a: np.ndarray, p_b: np.ndarray, n: int, seed: int
) -> dict[str, float]:
    """Bootstrap the AUC difference between two models scored on the same loans.

    Each resample draws loans with replacement and scores both models on the
    same draw, so noise shared by both cancels in the difference.

    Parameters
    ----------
    y : pd.Series
        Observed outcomes, 0 or 1.
    p_a, p_b : np.ndarray
        Predictions of model A and model B.
    n : int
        Number of resamples.
    seed : int
        Random seed.

    Returns
    -------
    dict of str to float
        ``difference`` (AUC of A minus B on all loans), ``low_95`` and
        ``high_95`` (bootstrap interval), and ``share_a_wins``.
    """
    y_arr = np.asarray(y)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y_arr), len(y_arr))
        diffs[i] = roc_auc_score(y_arr[idx], p_a[idx]) - roc_auc_score(
            y_arr[idx], p_b[idx]
        )
    return {
        "difference": float(roc_auc_score(y_arr, p_a) - roc_auc_score(y_arr, p_b)),
        "low_95": float(np.percentile(diffs, 2.5)),
        "high_95": float(np.percentile(diffs, 97.5)),
        "share_a_wins": float(np.mean(diffs > 0)),
    }
