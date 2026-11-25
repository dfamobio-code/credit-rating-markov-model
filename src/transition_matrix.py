"""
transition_matrix.py

Purpose:
    Construct our baseline Markov transition matrix using
    S&P Global's published credit-rating transition data.

Input:
    data/raw/sp_transition_matrix.csv

Output:
    data/processed/baseline_matrix.csv

Run:
    python src/transition_matrix.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. FILE PATHS AND CONFIGURATION
# ============================================================

# Locate the root directory of our repository.
# Assumes this script is located in the src/ folder.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Location of the original S&P transition matrix.
RAW_CSV = REPO_ROOT / "data" / "raw" / "sp_transition_matrix.csv"

# Location where our processed baseline matrix will be saved.
OUTPUT_CSV = REPO_ROOT / "data" / "processed" / "baseline_matrix.csv"

# The eight credit-rating states used in our Markov model.
# The order matters because it determines the rows and columns.
RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC/C", "D"]

# All credit ratings excluding default.
NON_DEFAULT_RATINGS = RATINGS[:-1]


# ============================================================
# 2. BUILD THE BASELINE TRANSITION MATRIX
# ============================================================

def build_transition_matrix(
    raw_path: Path = RAW_CSV,
    output_path: Path = OUTPUT_CSV,
) -> pd.DataFrame:

    """
    Convert S&P's published transition matrix into our
    baseline Markov transition matrix.

    Steps:
        1. Load the raw CSV.
        2. Validate the input data.
        3. Remove the NR column.
        4. Normalize each row (Option A).
        5. Add an absorbing default state.
        6. Validate the final matrix.
        7. Save the matrix as a CSV.

    Returns:
        An 8 x 8 pandas DataFrame containing probabilities.
    """

    # --------------------------------------------------------
    # STEP 1: LOAD THE RAW TRANSITION MATRIX
    # --------------------------------------------------------

    # Read S&P's published transition probabilities.
    # These are percentages, e.g., 87.28 represents 87.28%.
    raw = pd.read_csv(raw_path)

    # Verify that the CSV contains all required columns.
    required_columns = {"from_rating", *RATINGS, "NR"}

    missing_columns = required_columns - set(raw.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    # Remove unnecessary whitespace from rating labels.
    raw["from_rating"] = raw["from_rating"].astype(str).str.strip()

    # S&P provides seven starting ratings, excluding default.
    # Check that each expected rating appears exactly once.
    if (
        raw["from_rating"].duplicated().any()
        or set(raw["from_rating"]) != set(NON_DEFAULT_RATINGS)
    ):
        raise ValueError("Unexpected or duplicate starting ratings.")


    # --------------------------------------------------------
    # STEP 2: ORGANIZE AND VALIDATE THE DATA
    # --------------------------------------------------------

    # Set the starting rating as the row index.
    # Arrange rows and columns in our chosen rating order.
    percentages = (
        raw.set_index("from_rating")
        .loc[NON_DEFAULT_RATINGS, RATINGS + ["NR"]]
        .apply(pd.to_numeric, errors="raise")
    )

    # Convert the DataFrame into a NumPy array for validation.
    values = percentages.to_numpy(dtype=float)

    # All probabilities must be finite and nonnegative.
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Invalid transition percentages.")

    # Since the raw values are percentages, each row should
    # sum to approximately 100.
    #
    # We allow a small tolerance because S&P rounds values.
    if not np.allclose(
        percentages.sum(axis=1).to_numpy(),
        100.0,
        atol=0.05,
        rtol=0
    ):
        raise ValueError("Raw transition rows do not sum to 100%.")


    # --------------------------------------------------------
    # STEP 3: REMOVE NR AND NORMALIZE 
    # --------------------------------------------------------

    # Remove the Not Rated (NR) destination column.
    retained = percentages[RATINGS]

    # Calculate the sum of each row after removing NR.
    row_totals = retained.sum(axis=1)

    # Prevent division by zero.
    if (row_totals <= 0).any():
        raise ValueError("Cannot normalize a row with zero total.")

    # Normalize each row so its probabilities sum to 1.
    #
    # Mathematical formula:
    #
    #     P_ij = p_ij / sum_k(p_ik)
    #
    # where k includes all destinations except NR.
    #
    # For example:
    #     AAA -> AAA = 87.28
    #     AAA -> NR  = 3.08
    #
    # The new AAA -> AAA probability is approximately:
    #
    #     87.28 / (100 - 3.08) = 0.9005
    #
    # Using the actual retained row total handles rounding.
    matrix = retained.div(row_totals, axis=0)


    # --------------------------------------------------------
    # STEP 4: MAKE DEFAULT AN ABSORBING STATE
    # --------------------------------------------------------

    # Once a company reaches default (D), it stays in default.
    #
    # Therefore:
    #
    #     P(D -> D) = 1
    #     P(D -> j) = 0 for every j != D

    # Create the default row with zeros.
    matrix.loc["D"] = 0.0

    # Set the probability of remaining in default to 1.
    matrix.loc["D", "D"] = 1.0

    # Ensure the matrix follows our chosen rating order.
    matrix = matrix.loc[RATINGS, RATINGS]


    # --------------------------------------------------------
    # STEP 5: VALIDATE THE FINAL TRANSITION MATRIX
    # --------------------------------------------------------

    # Our model has eight states, so P must be 8 x 8.
    if matrix.shape != (8, 8):
        raise ValueError("Expected an 8 x 8 transition matrix.")

    # Ensure there are no NaN or infinite values.
    if not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("Matrix contains invalid numerical values.")

    # Every transition probability must be between 0 and 1.
    if (
        (matrix.to_numpy() < 0).any()
        or (matrix.to_numpy() > 1).any()
    ):
        raise ValueError("Probabilities must be between 0 and 1.")

    # Every row of a stochastic matrix must sum to 1.
    # np.allclose accounts for floating-point rounding errors.
    if not np.allclose(
        matrix.sum(axis=1).to_numpy(),
        1.0,
        atol=1e-12
    ):
        raise ValueError("Transition matrix rows must sum to 1.")


    # --------------------------------------------------------
    # STEP 6: SAVE THE PROCESSED MATRIX
    # --------------------------------------------------------

    # Create the processed data folder if it doesn't exist.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the final 8 x 8 transition matrix.
    # Values are now probabilities rather than percentages.
    matrix.to_csv(
        output_path,
        index_label="from_rating",
        float_format="%.12f"
    )

    # Return the matrix so other Python files can import
    # this function and use the resulting DataFrame.
    return matrix


# ============================================================
# 3. RUN THE SCRIPT
# ============================================================

# Only execute this block when the script is run directly.
# It will not execute when imported into another Python file.
if __name__ == "__main__":

    # Construct and save our baseline transition matrix.
    transition_matrix = build_transition_matrix()

    # Confirm the output location.
    print(f"Saved baseline matrix to: {OUTPUT_CSV}")

    # Display the matrix, rounded to six decimal places.
    print(transition_matrix.round(6).to_string())