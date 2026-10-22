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


def qubit_sweep(perf: pd.DataFrame, diag: pd.DataFrame, path):
    """Left: test ROC-AUC vs qubits. Right: kernel concentration vs qubits."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    summary = perf.groupby(["n_qubits", "model"])["roc_auc"].agg(["mean", "std"]).reset_index()
    for model, g in summary.groupby("model"):
        style = {"lw": 2.5} if model == "qsvc" else {"lw": 1.5, "alpha": 0.85}
        a.errorbar(g["n_qubits"], g["mean"], yerr=g["std"], marker="o", capsize=3,
                   label=LABELS.get(model, model), **style)
    a.set_xlabel("qubits (= PCA components)")
    a.set_ylabel("test ROC-AUC (scaffold split)")
    a.legend(frameon=False)
    a.grid(alpha=0.3)

    for bw, g in diag.groupby("bandwidth"):
        b.plot(g["n_qubits"], g["offdiag_var"], marker="o", label=f"bandwidth {bw:g}")
    b.set_yscale("log")
    b.set_xlabel("qubits")
    b.set_ylabel("variance of off-diagonal kernel values")
    b.set_title("kernel concentration", fontsize=10)
    b.legend(frameon=False)
    b.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def noise_study(results: pd.DataFrame, path):
    """Left: kernel error vs shots. Right: test ROC-AUC vs shots. One line per noise/reps."""
    raw = results[~results["psd_repair"] & (results["noise"] != "exact")]
    exact = results[results["noise"] == "exact"].groupby("reps")["roc_auc"].mean()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    for (noise, reps), g in raw.groupby(["noise", "reps"]):
        m = g.groupby("shots")[["rel_frobenius", "roc_auc"]].agg(["mean", "std"])
        label = f"{noise}, reps={reps}"
        a.errorbar(m.index, m[("rel_frobenius", "mean")], yerr=m[("rel_frobenius", "std")],
                   marker="o", capsize=3, label=label)
        b.errorbar(m.index, m[("roc_auc", "mean")], yerr=m[("roc_auc", "std")],
                   marker="o", capsize=3, label=label)
    for reps, auc in exact.items():
        b.axhline(auc, ls="--", lw=1, color="grey")
        b.annotate(f"exact kernel, reps={reps}", (raw["shots"].min(), auc), fontsize=8,
                   va="bottom")
    for ax in (a, b):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("shots per kernel entry")
        ax.grid(alpha=0.3)
    a.set_ylabel("relative error vs exact kernel")
    b.set_ylabel("test ROC-AUC")
    a.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
