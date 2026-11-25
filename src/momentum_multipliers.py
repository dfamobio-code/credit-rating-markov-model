"""Estimate 12 rating momentum multipliers from Bangia et al., Tables 4.2–4.3.

The inputs are four manually extracted CSVs of annual transition probabilities.
No network requests or data processing are performed at import time.
"""

import numpy as np
import pandas as pd

# The paper's rating scale, ordered from best to worst.
RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC/C", "D"]
COMMON_ORIGINS = ["AA", "A", "BBB", "BB", "B"]
EVENTS = ["upgrade", "unchanged", "downgrade", "default"]
MOMENTUM_STATES = ["up", "flat", "down"]


def load_momentum_matrix(csv_path):
    """Read one extracted CSV, with the origin ratings in its first column.

    Use decimal probabilities (e.g. 0.9181). Encode individual dashes in an
    observed row as 0; leave completely unobserved rows blank or omit them.
    Do not include the paper's issuer-count column or average-default footer.
    """
    return pd.read_csv(csv_path, index_col=0)


def load_momentum_matrices(unconditional_path, up_path, flat_path, down_path):
    """Return the unconditional reference and the three conditional matrices."""
    paths = {"unconditional": unconditional_path, "up": up_path,
             "flat": flat_path, "down": down_path}
    return {name: load_momentum_matrix(path) for name, path in paths.items()}


def transition_event_rates(matrix):
    """Collapse each common origin row into four mutually exclusive events.

    Keep ALL destination grades, including AAA and CCC, when summing
    upgrades and downgrades. Default is counted separately from downgrades.
    Missing rows outside COMMON_ORIGINS are deliberately not used.
    """
    matrix = matrix.copy()
    matrix.index = matrix.index.astype(str).str.strip().str.upper()
    matrix.columns = matrix.columns.astype(str).str.strip().str.upper()
    if not matrix.index.is_unique or not matrix.columns.is_unique:
        raise ValueError("Rating labels must be unique.")
    if set(matrix.columns) != set(RATINGS):
        raise ValueError(f"Expected exactly these destination ratings: {RATINGS}")
    if not set(COMMON_ORIGINS).issubset(matrix.index):
        raise ValueError(f"Missing one or more origin ratings: {COMMON_ORIGINS}")

    # Restrict origins, not destinations. Reject missing data in the selected rows.
    rows = matrix.loc[COMMON_ORIGINS, RATINGS].apply(pd.to_numeric, errors="raise")
    values = rows.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Selected rows must contain finite decimal probabilities in [0, 1].")
    sums = rows.sum(axis=1)
    if not np.allclose(sums, 1, rtol=0, atol=0.001):
        raise ValueError("Selected rows must sum to 1, up to published rounding.")

    # Correct minor rounding errors in the published probabilities.
    rows = rows.div(sums, axis=0)
    rates = pd.DataFrame(index=COMMON_ORIGINS, columns=EVENTS, dtype=float)
    for rating in COMMON_ORIGINS:
        i = RATINGS.index(rating)
        rates.loc[rating, "upgrade"] = rows.loc[rating, RATINGS[:i]].sum()
        rates.loc[rating, "unchanged"] = rows.loc[rating, rating]
        rates.loc[rating, "downgrade"] = rows.loc[rating, RATINGS[i + 1:-1]].sum()
        rates.loc[rating, "default"] = rows.loc[rating, "D"]
    return rates


def estimate_momentum_multipliers(matrices):
    """Return a 3 x 4 DataFrame of beta[m, event] multipliers.

    beta[m, c] = mean_i Pr(c | i, m) / mean_i Pr(c | i),
    where the mean gives each of the five common origin grades equal weight.
    The denominator is the PAPER'S unconditional matrix, not our S&P baseline.
    """
    reference = transition_event_rates(matrices["unconditional"]).mean(axis=0)
    if (reference <= 0).any():
        raise ValueError("Cannot estimate ratios with zero unconditional event rates.")

    multipliers = pd.DataFrame(index=MOMENTUM_STATES, columns=EVENTS, dtype=float)
    for momentum in MOMENTUM_STATES:
        conditional = transition_event_rates(matrices[momentum]).mean(axis=0)
        multipliers.loc[momentum] = conditional / reference
    multipliers.index.name = "momentum"
    return multipliers
