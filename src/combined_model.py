"""Construct combined regime- and momentum-dependent transition matrices."""

import numpy as np
import pandas as pd

# Ordered from strongest credit rating to default.
RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC/C", "D"]
CATEGORIES = ["upgrade", "unchanged", "downgrade", "default"]
REGIMES = ["expansion", "recession"]
MOMENTA = ["up", "flat", "down"]


def matrix_frame(matrix):
    """Convert a probability matrix to a labeled DataFrame."""

    n = len(RATINGS)

    if isinstance(matrix, pd.DataFrame):
        frame = matrix.copy()

        # Clean row and column labels
        frame.index = frame.index.map(lambda x: str(x).strip().upper())
        frame.columns = frame.columns.map(lambda x: str(x).strip().upper())

        # Reject duplicate rating labels
        if not frame.index.is_unique or not frame.columns.is_unique:
            raise ValueError("Matrix contains duplicate labels.")

        # Reorder matrix according to the model's rating states
        if set(frame.index) == set(RATINGS) and set(frame.columns) == set(RATINGS):
            return frame.loc[RATINGS, RATINGS].astype(float)

        # Only accept positional labels if rating labels were not supplied
        expected_labels = list(map(str, range(n)))

        if list(frame.index) != expected_labels or list(frame.columns) != expected_labels:
            raise ValueError("Matrix labels do not match RATINGS.")

    # Convert matrix to a NumPy array
    values = np.asarray(matrix, dtype=float)

    # Validate matrix dimensions
    if values.shape != (n, n):
        raise ValueError(f"Expected a {n} x {n} transition matrix.")

    # Return DataFrame with standardized rating labels
    return pd.DataFrame(values, index=RATINGS, columns=RATINGS)


def canonical_table(multipliers, states):
    """Return multiplier table with conditioning states as rows and categories as columns."""
    aliases = {"upward": "up", "downward": "down", "no change": "unchanged",
               "stay": "unchanged", "same": "unchanged", "stable": "unchanged",
               "upgrades": "upgrade", "downgrades": "downgrade", "defaults": "default"}
    def clean(label):
        label = str(label).strip().lower().replace("_multiplier", "")
        return aliases.get(label, label)

    frame = pd.DataFrame(multipliers).copy()
    frame.index = frame.index.map(clean)
    frame.columns = frame.columns.map(clean)
    # Accept either orientation and common long-format tables.
    for key in ("regime", "momentum", "state"):
        if key in frame.columns:
            if {"category", "multiplier"}.issubset(frame.columns):
                frame = frame.pivot(index=key, columns="category", values="multiplier")
            else:
                frame = frame.set_index(key)
            frame.index = frame.index.map(clean)
            frame.columns = frame.columns.map(clean)
            break
    if set(states).issubset(frame.columns) and set(CATEGORIES).issubset(frame.index):
        frame = frame.T
    if "flat" in states and "unchanged" in frame.index:
        frame = frame.rename(index={"unchanged": "flat"})
    if not set(states).issubset(frame.index) or not set(CATEGORIES).issubset(frame.columns):
        raise ValueError(f"Expected multiplier rows {states} and columns {CATEGORIES}; got {list(frame.index)} and {list(frame.columns)}.")
    frame = frame.loc[states, CATEGORIES].astype(float)
    if not np.isfinite(frame.to_numpy()).all() or (frame < 0).any().any():
        raise ValueError("Multipliers must be finite and nonnegative.")
    return frame


def combine_multipliers(alpha, beta):
    """Calculate gamma[e,m,c] = alpha[e,c] * beta[m,c]."""
    alpha = canonical_table(alpha, REGIMES)
    beta = canonical_table(beta, MOMENTA)
    index = pd.MultiIndex.from_product([REGIMES, MOMENTA], names=["regime", "momentum"])
    return pd.DataFrame([alpha.loc[e] * beta.loc[m] for e, m in index],
                        index=index, columns=CATEGORIES)


def build_combined_matrices(P, combined_multipliers):
    """Apply the combined weights and normalize each row of all six matrices."""
    P = matrix_frame(P)
    result = {}
    for e in REGIMES:
        for m in MOMENTA:
            weights = combined_multipliers.loc[(e, m), CATEGORIES]
            A = P.copy()
            # Classify each possible transition by its direction.
            for i, current in enumerate(RATINGS[:-1]):
                for j, target in enumerate(RATINGS):
                    category = ("default" if target == "D" else
                                "upgrade" if j < i else
                                "downgrade" if j > i else "unchanged")
                    A.loc[current, target] *= weights[category]
            # Default remains absorbing regardless of the multipliers.
            A.loc["D"] = 0.0
            A.loc["D", "D"] = 1.0
            totals = A.sum(axis=1)
            if (totals <= 0).any() or not np.isfinite(A.to_numpy()).all() or (A < 0).any().any():
                raise ValueError(f"Invalid weighted matrix for {(e, m)}.")
            result[e, m] = A.div(totals, axis=0)
    return result
