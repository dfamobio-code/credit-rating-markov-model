"""Analytical and Monte Carlo credit risk metrics."""

import numpy as np
import pandas as pd
from .combined_model import RATINGS, matrix_frame


def validate_matrix(P, atol=1e-8):
    """Check nonnegativity, unit row sums and an absorbing default state."""
    A = matrix_frame(P).to_numpy()
    if not np.isfinite(A).all() or (A < 0).any():
        raise ValueError("Transition probabilities must be finite and nonnegative.")
    if not np.allclose(A.sum(axis=1), 1, atol=atol, rtol=0):
        raise ValueError("Transition matrix rows must sum to one.")
    target = np.zeros(len(RATINGS))
    target[-1] = 1
    if not np.allclose(A[-1], target, atol=atol, rtol=0):
        raise ValueError("Default must be absorbing.")
    return True


def fixed_default_probabilities(P, horizons):
    """Return cumulative default probabilities when P is fixed over time."""
    validate_matrix(P)
    A = matrix_frame(P).to_numpy()
    return pd.DataFrame({h: np.linalg.matrix_power(A, h)[:, -1]
                         for h in horizons}, index=RATINGS)


def default_curve(K, initial, default_mask, time_horizon):
    """Calculate exact cumulative default probability from year zero onward."""
    K = np.asarray(K, dtype=float)
    mask = np.asarray(default_mask, dtype=bool)
    distribution = np.zeros(len(K))
    distribution[initial] = 1.0
    curve = np.empty(time_horizon + 1)
    for t in range(time_horizon + 1):
        curve[t] = distribution[mask].sum()
        distribution = distribution @ K
    return np.clip(curve, 0, 1)


def expected_times(K, default_mask):
    """Solve (I-Q)t=1 for states that reach default almost surely.

    States with a positive probability of never defaulting have infinite
    unconditional expected time to default.
    """
    K = np.asarray(K, dtype=float)
    default = np.asarray(default_mask, dtype=bool)
    edges = K > 0
    # Find all states from which default can be reached.
    reachable = default.copy()
    while True:
        expanded = reachable | edges[:, reachable].any(axis=1)
        if np.array_equal(expanded, reachable):
            break
        reachable = expanded
    # Any state that can enter a non-default closed class has infinite mean.
    bad = ~reachable
    while True:
        expanded = bad | (edges[:, bad].any(axis=1) & ~default)
        if np.array_equal(expanded, bad):
            break
        bad = expanded
    good = ~bad & ~default
    times = np.full(len(K), np.inf)
    times[default] = 0.0
    if good.any():
        Q = K[np.ix_(good, good)]
        times[good] = np.linalg.solve(np.eye(len(Q)) - Q, np.ones(len(Q)))
    return times


def empirical_curve(paths):
    """Calculate the fraction of simulated firms in default at each year."""
    paths = np.asarray(paths)
    return (paths == RATINGS.index("D")).mean(axis=0)


def time_to_default(paths):
    """Return first default year per firm; infinity denotes right-censoring."""
    hits = np.asarray(paths) == RATINGS.index("D")
    times = hits.argmax(axis=1).astype(float)
    times[~hits.any(axis=1)] = np.inf
    return times
