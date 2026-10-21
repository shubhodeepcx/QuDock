"""Figures for the benchmark results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

LABELS = {"qsvc": "QSVC (ZZ kernel)", "svm_rbf": "SVM (RBF)",
          "random_forest": "Random Forest", "logreg": "Logistic Regression"}


def learning_curve(summary: pd.DataFrame, path, metric: str = "ROC-AUC"):
    """Mean +/- std of the metric against training-set size, one line per model."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for model, g in summary.groupby("model"):
        g = g.sort_values("n_train")
        style = {"lw": 2.5} if model == "qsvc" else {"lw": 1.5, "alpha": 0.85}
        ax.plot(g["n_train"], g["mean"], marker="o", label=LABELS.get(model, model), **style)
        ax.fill_between(g["n_train"], g["mean"] - g["std"], g["mean"] + g["std"], alpha=0.15)
    ax.set_xscale("log")
    ax.set_xlabel("training molecules")
    ax.set_ylabel(f"test {metric} (scaffold split)")
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
