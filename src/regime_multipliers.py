"""Estimate regime multipliers from historical annual credit rating activity."""

import numpy as np
import pandas as pd

EVENTS = ["upgrade", "unchanged", "downgrade", "default"]
REGIMES = ["expansion", "recession"]


def load_rating_activity(csv_path):
    """Read annual S&P rates: year, upgrade, unchanged, downgrade, default.

    Use a consistent unit throughout (percentages or decimal probabilities).
    Downgrades must exclude defaults; withdrawn ratings are not modeled here.
    """
    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip().str.lower()
    return data[["year", *EVENTS]]


def merge_historical_data(rating_activity, annual_regimes):
    """Join S&P annual activity to FRED annual regime labels by year."""
    ratings = rating_activity[["year", *EVENTS]].copy()
    regimes = annual_regimes[["year", "regime"]].copy()
    for frame in (ratings, regimes):
        years = pd.to_numeric(frame["year"], errors="raise")
        if not np.isfinite(years).all() or (years % 1 != 0).any():
            raise ValueError("Years must be finite integers.")
        frame["year"] = years.astype(int)

    # Keep every rating year; fail if its economic regime is unavailable.
    merged = ratings.merge(regimes, on="year", how="left", validate="one_to_one")
    if merged["regime"].isna().any():
        missing = merged.loc[merged["regime"].isna(), "year"].tolist()
        raise ValueError(f"Missing FRED regime labels for years: {missing}")
    return merged.sort_values("year").reset_index(drop=True)


def estimate_regime_multipliers(merged_data):
    """Compute w_k(e) = mean(rate_k | regime e) / mean(rate_k).

    Each calendar year receives equal weight. The unconditional mean uses
    the same historical sample as the conditional means. The result has
    regimes as rows and the four transition types as columns.
    """
    data = merged_data[["year", "regime", *EVENTS]].copy()
    if data["year"].duplicated().any():
        raise ValueError("Expected exactly one observation per calendar year.")
    if set(data["regime"]) != set(REGIMES):
        raise ValueError("Both expansion and recession observations are required.")
    data[EVENTS] = data[EVENTS].apply(pd.to_numeric, errors="raise")
    rates = data[EVENTS].to_numpy(dtype=float)
    if not np.isfinite(rates).all() or (rates < 0).any() or (rates > 100).any():
        raise ValueError("Rates must be finite, nonnegative, and at most 100.")

    overall_means = data[EVENTS].mean()
    if (overall_means == 0).any():
        raise ValueError("Cannot estimate a multiplier for an event with zero overall rate.")

    # Divide each regime's mean rate by its unconditional mean rate.
    regime_means = data.groupby("regime")[EVENTS].mean()
    multipliers = regime_means.div(overall_means, axis="columns")
    return multipliers.loc[REGIMES, EVENTS]
