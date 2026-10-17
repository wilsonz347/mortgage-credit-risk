"""Simulated loans with known survival behavior, for tests."""

import numpy as np
import pandas as pd

from mortgage_credit_risk.events import DEFAULT, NO_EVENT


def simulated_survival(n: int, seed: int) -> pd.DataFrame:
    """Loans whose monthly default risk halves per standard deviation of score."""
    rng = np.random.default_rng(seed)
    loans = pd.DataFrame(
        {
            "credit_score": rng.normal(740, 40, n),
            "ltv": rng.normal(75, 15, n),
            "dti": rng.normal(35, 8, n),
            "orig_rate": rng.normal(4, 0.5, n),
            "purpose": rng.choice(["C", "N", "P"], n),
            "occupancy": rng.choice(["I", "P", "S"], n),
        }
    )
    hazard = 0.01 * np.exp(np.log(0.5) * (loans["credit_score"] - 740) / 40)
    default_age = rng.geometric(hazard)
    censor_age = rng.integers(1, 48, n)
    loans["last_age"] = np.minimum(default_age, censor_age)
    loans["event"] = np.where(default_age <= censor_age, DEFAULT, NO_EVENT)
    return loans
