def standardize_rating_labels(matrix):
    """Standardize credit rating labels across data sources."""

    return matrix.rename(
        index={"CCC": "CCC/C"},
        columns={"CCC": "CCC/C"}
    )