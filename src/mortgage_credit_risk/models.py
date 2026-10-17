"""Baseline models: fixed-horizon logistic regression and Cox."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

from mortgage_credit_risk.config import Config
from mortgage_credit_risk.events import DEFAULT, NO_EVENT


def horizon_label(loans: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Keep loans with a known answer at ``horizon`` and add a 0/1 target.

    A loan has a known answer if it was watched up to loan age ``horizon``,
    or if it defaulted or prepaid before then. A loan censored earlier is
    dropped: we cannot tell whether it would have defaulted in time. A loan
    that prepaid early counts as 0, since a paid-off loan cannot default.

    Parameters
    ----------
    loans : pd.DataFrame
        Loan table with ``last_age`` and ``event``.
    horizon : int
        Loan age in months.

    Returns
    -------
    pd.DataFrame
        Kept loans with a new ``target`` column: 1 if the loan defaulted by
        ``horizon``, else 0.
    """
    known = (loans["last_age"] >= horizon) | (loans["event"] != NO_EVENT)
    out = loans[known].copy()
    defaulted_in_time = (out["event"] == DEFAULT) & (out["last_age"] <= horizon)
    out["target"] = defaulted_in_time.astype(int)
    return out


def preprocessor(
    numeric: tuple[str, ...], categorical: tuple[str, ...]
) -> ColumnTransformer:
    """Build the feature preprocessing step. Fit it on training data only.

    Numeric features: missing values are filled with the training median,
    then each feature is rescaled to mean 0 and standard deviation 1 using
    training statistics. Categorical features: each category becomes a 0/1
    column. The first category of each feature is dropped as the reference,
    and categories not seen in training become all zeros.

    Parameters
    ----------
    numeric : tuple of str
        Numeric columns.
    categorical : tuple of str
        Categorical columns.

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessing step.
    """
    return ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                list(numeric),
            ),
            (
                "categorical",
                OneHotEncoder(drop="first", handle_unknown="ignore"),
                list(categorical),
            ),
        ],
        verbose_feature_names_out=False,
    )


def fit_logistic(train: pd.DataFrame, cfg: Config) -> Pipeline:
    """Fit preprocessing and logistic regression on training loans.

    Classes are not reweighted, so predicted probabilities stay on the real
    scale of default rates.

    Parameters
    ----------
    train : pd.DataFrame
        Training loans from ``horizon_label``, with ``target``.
    cfg : Config
        Project settings: feature lists, seed and model parameters.

    Returns
    -------
    Pipeline
        Fitted preprocessing and model. ``predict_proba(X)[:, 1]`` is the
        probability of default by the horizon.
    """
    features = [*cfg.numeric_features, *cfg.categorical_features]
    model = Pipeline(
        [
            ("pre", preprocessor(cfg.numeric_features, cfg.categorical_features)),
            (
                "logit",
                LogisticRegression(
                    class_weight=None, random_state=cfg.seed, **cfg.logistic_params
                ),
            ),
        ]
    )
    return model.fit(train[features], train["target"])


def coefficients(model: Pipeline) -> pd.DataFrame:
    """List the fitted coefficients and odds ratios.

    Parameters
    ----------
    model : Pipeline
        Output of ``fit_logistic``.

    Returns
    -------
    pd.DataFrame
        One row per input column plus the intercept, with ``coef`` (change in
        log odds per unit of the scaled input) and ``odds_ratio`` (``exp(coef)``).
    """
    logit = model.named_steps["logit"]
    names = ["intercept", *model.named_steps["pre"].get_feature_names_out()]
    coef = [logit.intercept_[0], *logit.coef_[0]]
    out = pd.DataFrame({"coef": coef}, index=pd.Index(names, name="feature"))
    out["odds_ratio"] = np.exp(out["coef"])
    return out


def survival_target(loans: pd.DataFrame) -> np.ndarray:
    """Build the duration and event target for survival models.

    Duration is the loan age in the last observed month. The event is
    default. Prepaid and still-open loans count as censored.

    Parameters
    ----------
    loans : pd.DataFrame
        Loan table with ``last_age`` and ``event``.

    Returns
    -------
    np.ndarray
        Structured array with fields ``event`` (bool) and ``time``.
    """
    target: np.ndarray = Surv.from_arrays(
        event=(loans["event"] == DEFAULT).to_numpy(),
        time=loans["last_age"].to_numpy(),
    )
    return target


def fit_cox(train: pd.DataFrame, cfg: Config) -> Pipeline:
    """Fit preprocessing and a Cox model for default on training loans.

    Uses the same preprocessing as ``fit_logistic``, fit on training data.

    Parameters
    ----------
    train : pd.DataFrame
        Training loan table with ``last_age``, ``event`` and the features.
    cfg : Config
        Project settings: feature lists and model parameters.

    Returns
    -------
    Pipeline
        Fitted preprocessing and model. ``predict(X)`` gives a risk score
        (higher means riskier).
    """
    features = [*cfg.numeric_features, *cfg.categorical_features]
    model = Pipeline(
        [
            ("pre", preprocessor(cfg.numeric_features, cfg.categorical_features)),
            ("cox", CoxPHSurvivalAnalysis(**cfg.cox_params)),
        ]
    )
    return model.fit(train[features], survival_target(train))


def hazard_ratios(model: Pipeline) -> pd.DataFrame:
    """List the fitted Cox coefficients and hazard ratios.

    Parameters
    ----------
    model : Pipeline
        Output of ``fit_cox``.

    Returns
    -------
    pd.DataFrame
        One row per input column, with ``coef`` and ``hazard_ratio``
        (``exp(coef)``): how much the monthly default risk is multiplied
        per unit of the scaled input, at every loan age.
    """
    names = model.named_steps["pre"].get_feature_names_out()
    coef = model.named_steps["cox"].coef_
    out = pd.DataFrame({"coef": coef}, index=pd.Index(names, name="feature"))
    out["hazard_ratio"] = np.exp(out["coef"])
    return out


def default_by(model: Pipeline, loans: pd.DataFrame, month: int) -> np.ndarray:
    """Predict each loan's chance of default by a given loan age.

    Computed as ``1 - S(month)``, where ``S`` is the loan's survival curve
    from the Cox model.

    Parameters
    ----------
    model : Pipeline
        Output of ``fit_cox``.
    loans : pd.DataFrame
        Loans with the model features.
    month : int
        Loan age. Must not be past the longest duration seen in training.

    Returns
    -------
    np.ndarray
        Probability of default by ``month`` for each loan.
    """
    curves = model.predict_survival_function(loans[model.feature_names_in_])
    return np.array([1.0 - curve(month) for curve in curves])
