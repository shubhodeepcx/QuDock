"""Figures for the benchmark results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
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


FINAL_LABELS = {
    "svm_rbf_pca": "SVM-RBF, PCA", "rf_pca": "Random Forest, PCA", "qsvc_pca": "QSVC, PCA",
    "qsvc_pls": "QSVC, PLS + tuned bandwidth", "projected_pls": "Projected quantum kernel, PLS",
    "tanimoto_svm": "Tanimoto SVM", "rf_fingerprint": "Random Forest, fingerprint",
    "gb_fingerprint": "Gradient boosting, fingerprint + descriptors",
    "svm_rbf_pls": "SVM-RBF, PLS", "rf_pls": "Random Forest, PLS",
    "hybrid_pca": "Hybrid: quantum (PCA) + Tanimoto",
    "hybrid_pls": "Hybrid: quantum (PLS) + Tanimoto",
}
QUANTUM = {"qsvc_pca", "qsvc_pls", "projected_pls", "hybrid_pca", "hybrid_pls"}


def final_study(summary: pd.DataFrame, path):
    """Test ROC-AUC (mean, 95% interval over seeds) against training size.

    Left: models on 6 compressed features. Right: models that see the full fingerprint.
    """
    panels = [("6 qubit-sized features", ["svm_rbf_pls", "rf_pls", "qsvc_pca", "qsvc_pls",
                                          "projected_pls"]),
              ("full 2,048-bit fingerprint", ["tanimoto_svm", "rf_fingerprint", "gb_fingerprint",
                                              "hybrid_pca", "hybrid_pls"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (title, models) in zip(axes, panels):
        for model in models:
            g = summary[summary["model"] == model].sort_values("n_train")
            if g.empty:
                continue
            style = ({"lw": 2.4} if model in QUANTUM else {"lw": 1.4, "ls": "--", "alpha": 0.9})
            ax.plot(g["n_train"], g["roc_auc"], marker="o", label=FINAL_LABELS[model], **style)
            ax.fill_between(g["n_train"], g["roc_auc"] - g["roc_auc_ci"],
                            g["roc_auc"] + g["roc_auc_ci"], alpha=0.12)
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("training molecules")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("test ROC-AUC (scaffold split)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def confidence_study(sets: pd.DataFrame, domain: pd.DataFrame, path):
    """Left: promised vs measured coverage.

    Right: ROC-AUC against similarity to the training set.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot([0.65, 0.95], [0.65, 0.95], color="0.4", ls=":", label="promised")
    for model, g in sets.groupby("model", sort=False):
        g = g.sort_values("confidence")
        style = {"lw": 2.4} if model in QUANTUM else {"lw": 1.4, "ls": "--", "alpha": 0.9}
        axes[0].plot(g["confidence"], g["coverage"], marker="o",
                     label=FINAL_LABELS.get(model, model), **style)
    axes[0].set_xlabel("confidence asked for")
    axes[0].set_ylabel("molecules whose true label is in the set")
    axes[0].set_title("Calibration under a scaffold split")

    width = 0.26
    centres = domain.groupby(["similarity_from", "similarity_to"], sort=True).ngroup().unique()
    for i, (model, g) in enumerate(domain.groupby("model", sort=False)):
        g = g.sort_values("similarity_from")
        axes[1].bar(centres + (i - 1) * width, g["roc_auc"], width,
                    label=FINAL_LABELS.get(model, model))
    labels = [f"{lo:.2f} to {hi:.2f}" for lo, hi in
              domain.drop_duplicates("similarity_from").sort_values("similarity_from")
              [["similarity_from", "similarity_to"]].values]
    axes[1].set_xticks(centres, labels)
    axes[1].set_ylim(0.5, 0.85)
    axes[1].set_xlabel("Tanimoto similarity to the nearest training molecule")
    axes[1].set_ylabel("test ROC-AUC")
    axes[1].set_title("Accuracy against how new the chemistry is")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def scaling_study(results: pd.DataFrame, path):
    """Left: the kernel's off-diagonal spread as qubits grow. Right: what it costs in ROC-AUC."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for kind, g in results.groupby("kind", sort=False):
        g = g.sort_values("n_qubits")
        axes[0].plot(g["n_qubits"], g["off_diagonal_spread"], marker="o", label=kind)
        axes[1].plot(g["n_qubits"], g["roc_auc"], marker="o", label=kind)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("spread of off-diagonal kernel values")
    axes[0].set_title("Kernel concentration")
    axes[1].set_ylabel("test ROC-AUC (scaffold split)")
    axes[1].set_title("What concentration costs")
    for ax in axes:
        ax.set_xlabel("qubits")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def trainable_study(summary: pd.DataFrame, path):
    """Left: test ROC-AUC of the fixed and trained feature maps. Right: their alignment."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    styles = {"fixed": ("#B8BDEB", "--"), "trained": ("#5B3BA8", "-")}
    for name, g in summary.groupby("feature_map", sort=False):
        g = g.sort_values("n_train")
        color, ls = styles.get(name, ("#333333", "-"))
        label = "ZZ map, fixed" if name == "fixed" else "ZZ map, parameters trained"
        axes[0].plot(g["n_train"], g["roc_auc"], marker="o", color=color, ls=ls, lw=2, label=label)
        axes[0].fill_between(g["n_train"], g["roc_auc"] - g["roc_auc_ci"],
                             g["roc_auc"] + g["roc_auc_ci"], color=color, alpha=0.15)
        axes[1].plot(g["n_train"], g["alignment"], marker="o", color=color, ls=ls, lw=2,
                     label=label)
    axes[0].set_ylabel("test ROC-AUC (scaffold split)")
    axes[0].set_title("What training the feature map buys")
    axes[1].set_ylabel("kernel-target alignment")
    axes[1].set_title("How well the kernel matches the labels")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("training molecules")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def scorecard(scorecard: pd.DataFrame, verdicts, path):
    """Best quantum model minus best classical model, one row per setting."""
    data = scorecard.dropna(subset=["delta"]).reset_index(drop=True)
    marks = verdicts.loc[data.index]
    colors = {"quantum wins": "#5B3BA8", "classical wins": "#E8639F", "tie": "#8A8FB9"}
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(data) + 1.6))
    y = np.arange(len(data))[::-1]
    ax.errorbar(data["delta"], y, xerr=data["ci"], fmt="o", ms=7, lw=0, elinewidth=1.6,
                ecolor="#B8BDEB", color="none")
    for yi, (delta, verdict) in enumerate(zip(data["delta"], marks)):
        ax.plot(delta, y[yi], "o", ms=7, color=colors[verdict])
    ax.axvline(0, color="#2E1B5B", lw=1)
    ax.set_yticks(y, data["setting"], fontsize=9)
    ax.set_xlabel("ROC-AUC: best quantum model minus best classical model")
    ax.set_title("Where the quantum side wins, ties or loses")
    ax.grid(alpha=0.3, axis="x")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k) for k, c in colors.items()]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def explain_study(results: pd.DataFrame, path):
    """Left: removing the highlighted atoms against removing as many random ones.
    Right: the gap per molecule, so the misses are visible too."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    wins = results["gap"] > 0
    lim = [min(results["random_atoms"].min(), results["top_atoms"].min()) - 0.1,
           max(results["random_atoms"].max(), results["top_atoms"].max()) + 0.1]
    a.plot(lim, lim, color="#8A8FB9", ls=":", lw=1)
    a.scatter(results.loc[wins, "random_atoms"], results.loc[wins, "top_atoms"], s=45,
              color="#5B3BA8", label="explanation matters more")
    a.scatter(results.loc[~wins, "random_atoms"], results.loc[~wins, "top_atoms"], s=45,
              color="#E8639F", label="random atoms matter more")
    a.set_xlabel("score lost when random atoms are removed")
    a.set_ylabel("score lost when the highlighted atoms are removed")
    a.set_title(f"The highlighted atoms carry the prediction ({wins.sum()}/{len(results)})")
    a.legend(frameon=False, fontsize=9, loc="upper left")

    order = results.sort_values("gap")
    y = np.arange(len(order))
    b.barh(y, order["gap"], color=np.where(order["gap"] > 0, "#5B3BA8", "#E8639F"))
    b.axvline(0, color="#2E1B5B", lw=1)
    b.set_yticks(y, [s[:18] for s in order["top_substructure"]], fontsize=8)
    b.set_xlabel("extra score lost, highlighted minus random")
    b.set_title("Per molecule, labelled by its top substructure")
    for ax in (a, b):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


GRAPH_LABELS = {"graph_quantum": "Pharmacophore-graph quantum kernel",
                "graph_rbf": "SVM-RBF, same graph numbers",
                "graph_rf": "Random Forest, same graph numbers",
                "qsvc_pls": "ZZ map on 6 PLS angles"}


def graph_study(results: pd.DataFrame, path):
    """Test ROC-AUC against training size for the graph map and its classical twins."""
    summary = results.groupby(["n_train", "model"])["roc_auc"].agg(["mean", "sem"]).reset_index()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for model, label in GRAPH_LABELS.items():
        g = summary[summary["model"] == model].sort_values("n_train")
        if g.empty:
            continue
        quantum = model in ("graph_quantum", "qsvc_pls")
        ax.errorbar(g["n_train"], g["mean"], yerr=1.96 * g["sem"].fillna(0), marker="o",
                    capsize=3, label=label, lw=2.2 if quantum else 1.3, ls="-" if quantum else "--")
    ax.set_xscale("log")
    ax.set_xlabel("training molecules")
    ax.set_ylabel("test ROC-AUC (scaffold split)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def robustness_study(summary: pd.DataFrame, spread: pd.DataFrame, path):
    """Left: the quantum-classical gap on each scaffold split. Right: each model's range."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    y = np.arange(len(summary))
    colors = np.where(summary["delta"] > 0, "#5B3BA8", "#E8639F")
    a.barh(y, summary["delta"], color=colors)
    a.axvline(0, color="#2E1B5B", lw=1)
    a.axvline(summary["delta"].mean(), color="#8A8FB9", ls="--", lw=1.2,
              label=f"mean {summary['delta'].mean():+.3f}")
    a.set_yticks(y, [f"split {s}" for s in summary["split_seed"]], fontsize=9)
    a.set_xlabel("ROC-AUC: best quantum model minus best classical model")
    a.set_title("One split is not an answer")
    a.legend(frameon=False, fontsize=9, loc="lower right")

    order = spread.sort_values("mean")
    y = np.arange(len(order))
    b.hlines(y, order["worst"], order["best"], color="#B8BDEB", lw=3)
    b.plot(order["mean"], y, "o", ms=7, color="#5B3BA8")
    b.set_yticks(y, order["model"], fontsize=9)
    b.set_xlabel("test ROC-AUC across splits (worst, mean, best)")
    b.set_title("How far each model moves between splits")
    for ax in (a, b):
        ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


CAMPAIGN_COLORS = {"hybrid_pls": "#5B3BA8", "tanimoto_svm": "#E8639F",
                   "rf_fingerprint": "#3AA67E"}
MODEL_NAMES = {"hybrid_pls": "Hybrid (quantum + Tanimoto)", "tanimoto_svm": "Tanimoto SVM",
               "rf_fingerprint": "Random Forest (fingerprint)", "qsvc_pls": "QSVC (PLS angles)",
               "projected_pls": "Projected kernel", "svm_rbf_pls": "SVM-RBF (PLS angles)",
               "rf_pls": "Random Forest (PLS angles)"}
CAMPAIGN_LINES = {"greedy": "-", "uncertainty": "--", "random": ":"}


def campaign_study(summary: pd.DataFrame, path):
    """Left: inhibitors found against molecules tested. Right: what the model learned."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    for (model, strategy), g in summary.groupby(["model", "strategy"]):
        g = g.sort_values("n_tested")
        style = {"color": CAMPAIGN_COLORS.get(model, "#333333"),
                 "ls": CAMPAIGN_LINES.get(strategy, "-"), "lw": 2}
        label = f"{MODEL_NAMES.get(model, model)}, {strategy}"
        a.plot(g["n_tested"], g["actives_found"], marker="o", ms=4, label=label, **style)
        b.plot(g["n_tested"], g["roc_auc"], marker="o", ms=4, label=label, **style)
        b.fill_between(g["n_tested"], g["roc_auc"] - g["roc_auc_ci"],
                       g["roc_auc"] + g["roc_auc_ci"], color=style["color"], alpha=0.1)
    a.set_ylabel("inhibitors found")
    a.set_title("What the campaign found")
    b.set_ylabel("test ROC-AUC (unseen scaffolds)")
    b.set_title("What the model learned")
    for ax in (a, b):
        ax.set_xlabel("molecules tested")
        ax.grid(alpha=0.3)
    a.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cost_study(results: pd.DataFrame, path):
    """Left: seconds to train and score. Right: device time the same run would need."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    for model, g in results.groupby("model"):
        g = g.sort_values("n_train")
        style = {"lw": 2.2 if g["kind"].iloc[0] != "classical" else 1.4,
                 "alpha": 1.0 if g["kind"].iloc[0] != "classical" else 0.8}
        a.plot(g["n_train"], g["total_seconds"], marker="o",
               label=MODEL_NAMES.get(model, model), **style)
        device = g[g["device_seconds"] > 0]
        if len(device):
            b.plot(device["n_train"], device["device_seconds"] / 3600, marker="o",
                   label=MODEL_NAMES.get(model, model), **style)
    a.set_ylabel("seconds, simulated on one laptop core")
    a.set_title("What it costs to run here")
    b.set_ylabel("hours of device time")
    b.set_title("What the same run would cost on hardware")
    b.axhline(20 / 3600, color="#8A8FB9", ls=":", lw=1.2)
    b.annotate("our 20 s hardware run", (results["n_train"].min(), 20 / 3600),
               textcoords="offset points", xytext=(4, 6), fontsize=8, color="#555555")
    for ax in (a, b):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("training molecules")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
