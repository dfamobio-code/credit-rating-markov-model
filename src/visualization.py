"""Visualize transition matrices and simulated credit rating trajectories."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .combined_model import RATINGS, matrix_frame


def plot_matrices(matrices):
    """Plot labeled transition probability heatmaps on a common scale."""
    n = len(matrices)
    if n == 0:
        raise ValueError("Provide at least one transition matrix.")
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.4 * rows), squeeze=False)
    for ax, (name, matrix) in zip(axes.flat, matrices.items()):
        sns.heatmap(matrix_frame(matrix), ax=ax, annot=True, fmt=".1%",
                    cmap="Blues", vmin=0, vmax=1, cbar=False)
        ax.set(title=str(name), xlabel="Next rating", ylabel="Current rating")
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    plt.show()
    return fig


def plot_paths(paths, n_paths=5):
    """Plot sample rating trajectories for each simulated model."""
    if not paths:
        raise ValueError("Provide at least one set of simulated paths.")
    n = len(paths)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.5 * rows), squeeze=False)
    for ax, (name, model_paths) in zip(axes.flat, paths.items()):
        model_paths = np.asarray(model_paths)
        for path in model_paths[:n_paths]:
            ax.step(np.arange(len(path)), path, where="post", alpha=0.8)
        ax.set(title=str(name), xlabel="Year", ylabel="Credit rating",
               yticks=np.arange(len(RATINGS)), yticklabels=RATINGS,
               ylim=(len(RATINGS) - 0.7, -0.3))
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    plt.show()
    return fig
