"""Apply economic multipliers to a baseline credit transition matrix."""

import numpy as np
import pandas as pd

# Best to worst. Supports the project's seven/eight-state variants and
# S&P's common combined CCC/C category, without imposing a matrix order.
RATING_ORDER = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CCC/C", "CC", "C", "D"]
EVENTS = ["upgrade", "unchanged", "downgrade", "default"]
REGIMES = ["expansion", "recession"]


def _validate_baseline(baseline):
    """Require matching rating labels and probability rows summing to one."""
    matrix = baseline.copy()
    matrix.index = matrix.index.astype(str).str.strip().str.upper()
    matrix.columns = matrix.columns.astype(str).str.strip().str.upper()
    if not matrix.index.is_unique or not matrix.columns.is_unique:
        raise ValueError("Rating labels must be unique.")
    if set(matrix.index) != set(matrix.columns) or "D" not in matrix.index:
        raise ValueError("Expected matching row/column rating labels, including D.")
    unknown = set(matrix.index) - set(RATING_ORDER)
    if unknown:
        raise ValueError(f"Unsupported rating states: {unknown}. Handle NR separately.")
    if "CCC/C" in matrix.index and any(s in matrix.index for s in ["CCC", "CC", "C"]):
        raise ValueError("Do not mix the combined CCC/C state with its components.")

    matrix = matrix.loc[:, matrix.index].astype(float)
    values = matrix.to_numpy()
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Baseline probabilities must be finite and nonnegative.")
    row_sums = matrix.sum(axis=1)
    # Accept percentages as well as probabilities, allowing small rounding errors.
    if np.allclose(row_sums, 100, rtol=0, atol=0.1):
        matrix = matrix / 100
    elif not np.allclose(row_sums, 1, rtol=0, atol=0.001):
        raise ValueError("Baseline rows must sum to 1 (or 100 for percentages).")
    matrix = matrix.div(matrix.sum(axis=1), axis=0)
    if not np.isclose(matrix.loc["D", "D"], 1, rtol=0, atol=1e-10):
        raise ValueError("Default (D) must be absorbing in the baseline.")
    matrix.loc["D", :] = 0.0
    matrix.loc["D", "D"] = 1.0
    return matrix


def load_baseline_matrix(csv_path):
    """Load a CSV whose first column contains the origin rating labels."""
    return _validate_baseline(pd.read_csv(csv_path, index_col=0))


def transition_type(current_rating, next_rating):
    """Identify the event associated with a non-default origin state."""
    if current_rating == "D":
        raise ValueError("The default origin is absorbing and is not adjusted.")
    if next_rating == "D":
        return "default"
    if next_rating == current_rating:
        return "unchanged"
    if RATING_ORDER.index(next_rating) < RATING_ORDER.index(current_rating):
        return "upgrade"
    return "downgrade"


def build_regime_matrix(baseline, multipliers):
    """Weight baseline entries by event, then normalize every row.

    P_e(i,j) = P(i,j) * w_e(i,j) / sum_k[P(i,k) * w_e(i,k)].
    multipliers is a mapping or Series indexed by the four event names.
    """
    baseline = _validate_baseline(baseline)
    weights = {event: float(multipliers[event]) for event in EVENTS}
    if any(not np.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError("Multipliers must be finite and nonnegative.")

    adjusted = baseline.copy()
    for current_rating in baseline.index:
        if current_rating == "D":
            continue  # Preserve the absorbing default row exactly.
        for next_rating in baseline.columns:
            event = transition_type(current_rating, next_rating)
            adjusted.loc[current_rating, next_rating] *= weights[event]

        row_sum = adjusted.loc[current_rating].sum()
        if row_sum <= 0:
            raise ValueError(f"All adjusted probabilities are zero for {current_rating}.")
        adjusted.loc[current_rating] /= row_sum
    return adjusted


def build_regime_matrices(baseline, multipliers):
    """Return a dictionary containing expansion and recession matrices."""
    return {regime: build_regime_matrix(baseline, multipliers.loc[regime])
            for regime in REGIMES}
