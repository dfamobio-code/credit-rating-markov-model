"""Build augmented Markov chains and simulate annual credit rating paths."""

from itertools import product
import numpy as np
import pandas as pd
from .combined_model import RATINGS, REGIMES, MOMENTA, matrix_frame
from .metrics import validate_matrix


def annual_regime_series(regimes):
    """Return a sorted Series of annual regime labels indexed by year."""
    if isinstance(regimes, pd.Series):
        series = regimes.copy()
    else:
        frame = pd.DataFrame(regimes).copy()
        frame.columns = frame.columns.map(lambda x: str(x).strip().lower())
        if "year" in frame.columns:
            frame = frame.set_index("year")
        col = next((c for c in ("regime", "economic_regime", "recession", "usrec") if c in frame), None)
        if col is None:
            raise ValueError("Expected a regime or recession column.")
        series = frame[col].copy()
    if isinstance(series.index, pd.DatetimeIndex):
        series.index = series.index.year
    else:
        series.index = pd.to_numeric(series.index, errors="raise").astype(int)
    series = series.map(lambda x: str(x).strip().lower())
    series = series.replace({"0": "expansion", "0.0": "expansion", "false": "expansion",
                             "1": "recession", "1.0": "recession", "true": "recession"})
    if series.index.has_duplicates or not series.isin(REGIMES).all():
        raise ValueError("Expected one valid regime label per year.")
    return series.sort_index().rename("regime")


def estimate_regime_transition(regimes):
    """Estimate G from consecutive observed annual regime transitions."""
    series = annual_regime_series(regimes)
    counts = pd.DataFrame(0.0, index=REGIMES, columns=REGIMES)
    for k in range(len(series) - 1):
        if series.index[k + 1] == series.index[k] + 1:
            counts.loc[series.iloc[k], series.iloc[k + 1]] += 1
    totals = counts.sum(axis=1)
    if (totals == 0).any():
        raise ValueError("Insufficient consecutive observations for both regimes.")
    return counts.div(totals, axis=0)


def build_chain(P, regime_matrices=None, momentum_matrices=None,
                combined_matrices=None, G=None):
    """Build a homogeneous transition matrix on the relevant augmented state space."""
    if sum(x is not None for x in (regime_matrices, momentum_matrices, combined_matrices)) > 1:
        raise ValueError("Choose only one conditional model.")
    has_regime = regime_matrices is not None or combined_matrices is not None
    has_momentum = momentum_matrices is not None or combined_matrices is not None
    es = REGIMES if has_regime else [None]
    ms = MOMENTA if has_momentum else [None]
    if has_regime:
        if G is None:
            raise ValueError("Regime-dependent chains require G.")
        if isinstance(G, pd.DataFrame):
            G = G.loc[REGIMES, REGIMES]
        G = np.asarray(G, dtype=float)
        if G.shape != (2, 2) or not np.isfinite(G).all() or (G < 0).any() or not np.allclose(G.sum(axis=1), 1):
            raise ValueError("G must be a stochastic 2 x 2 matrix.")
    matrices = {}
    for e, m in product(es, ms):
        raw = (combined_matrices[e, m] if combined_matrices is not None else
               regime_matrices[e] if regime_matrices is not None else
               momentum_matrices[m] if momentum_matrices is not None else P)
        frame = matrix_frame(raw)
        validate_matrix(frame)
        matrices[e, m] = frame.to_numpy()

    # Keep one absorbing default state, rather than duplicating it across regimes.
    states = list(product(RATINGS[:-1], es, ms)) + [("D", None, None)]
    lookup = {state: i for i, state in enumerate(states)}
    K = np.zeros((len(states), len(states)))
    default = len(states) - 1
    for k, (r, e, m) in enumerate(states[:-1]):
        i = RATINGS.index(r)
        for j, target in enumerate(RATINGS):
            p = matrices[e, m][i, j]
            if target == "D":
                K[k, default] += p
                continue
            new_m = ("up" if j < i else "down" if j > i else "flat") if has_momentum else None
            # Rating and next regime are conditionally independent given current regime.
            for new_e in es:
                g = G[REGIMES.index(e), REGIMES.index(new_e)] if has_regime else 1.0
                K[k, lookup[target, new_e, new_m]] += p * g
    K[default, default] = 1.0
    return {"K": K, "states": states, "lookup": lookup,
            "default_mask": np.array([s[0] == "D" for s in states]),
            "rating_index": np.array([RATINGS.index(s[0]) for s in states]),
            "has_regime": has_regime, "has_momentum": has_momentum}


def initial_index(chain, rating, regime="expansion", momentum="flat"):
    """Get the index of an initial credit state in a particular chain."""
    state = ("D", None, None) if rating == "D" else (
        rating, regime if chain["has_regime"] else None,
        momentum if chain["has_momentum"] else None)
    return chain["lookup"][state]


def simulate_paths(chain, initial, n_simulations, time_horizon, seed=None):
    """Simulate augmented states; return paths of rating indices, including year zero."""
    if n_simulations <= 0 or time_horizon < 0:
        raise ValueError("Use a positive number of paths and nonnegative horizon.")
    rng = np.random.default_rng(seed)
    K = chain["K"]
    current = np.full(n_simulations, initial, dtype=int)
    paths = np.empty((n_simulations, time_horizon + 1), dtype=int)
    paths[:, 0] = chain["rating_index"][current]
    for t in range(1, time_horizon + 1):
        next_states = np.empty_like(current)
        for state in np.unique(current):
            mask = current == state
            p = K[state] / K[state].sum()
            next_states[mask] = rng.choice(len(K), size=mask.sum(), p=p)
        current = next_states
        paths[:, t] = chain["rating_index"][current]
    return paths
