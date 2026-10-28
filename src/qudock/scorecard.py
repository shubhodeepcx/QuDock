"""One chart that answers the only question judges really ask: does quantum help?

Every setting we measured becomes one row: the best quantum-containing model
against the best purely classical model on the same molecules, with a 95%
interval over seeds. Positive means the quantum side won. Settings come from
the result files, so the scorecard cannot drift from the experiments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

QUANTUM = {"qsvc_pca", "qsvc_pls", "projected_pls", "hybrid_pca", "hybrid_pls"}
CLASSICAL = {"svm_rbf_pca", "svm_rbf_pls", "rf_pca", "rf_pls", "tanimoto_svm", "rf_fingerprint"}


def _best_pair(runs: pd.DataFrame, setting: str) -> dict:
    """Best quantum and best classical mean ROC-AUC, and the gap between them."""
    means = runs.groupby("model")["roc_auc"].agg(["mean", "sem", "count"])
    quantum = means.loc[means.index.isin(QUANTUM)]
    classical = means.loc[means.index.isin(CLASSICAL)]
    if quantum.empty or classical.empty:
        return {}
    q, c = quantum["mean"].idxmax(), classical["mean"].idxmax()
    delta = quantum.loc[q, "mean"] - classical.loc[c, "mean"]
    spread = np.hypot(quantum.loc[q, "sem"], classical.loc[c, "sem"])
    return {"setting": setting, "quantum_model": q, "classical_model": c,
            "quantum": quantum.loc[q, "mean"], "classical": classical.loc[c, "mean"],
            "delta": delta, "ci": 1.96 * (0.0 if np.isnan(spread) else spread),
            "seeds": int(quantum.loc[q, "count"])}


def build_scorecard(final_runs: pd.DataFrame, bbbp_runs: pd.DataFrame | None = None,
                    scaling: pd.DataFrame | None = None,
                    trainable: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for size, g in final_runs.groupby("n_train"):
        rows.append(_best_pair(g, f"BACE, {size:,} molecules"))
    if bbbp_runs is not None:
        full = bbbp_runs[bbbp_runs["n_train"] == bbbp_runs["n_train"].max()]
        rows.append(_best_pair(full, f"BBBP, {full['n_train'].max():,} molecules"))
    if scaling is not None:
        for qubits, g in scaling.groupby("n_qubits"):
            best = g.loc[g["roc_auc"].idxmax()]
            rows.append({"setting": f"BACE, {qubits} qubits (no classical arm)",
                         "quantum_model": best["kind"], "classical_model": "",
                         "quantum": best["roc_auc"], "classical": np.nan,
                         "delta": np.nan, "ci": 0.0, "seeds": 1})
    if trainable is not None:
        pivot = trainable.pivot_table(index=["n_train", "seed"], columns="feature_map",
                                      values="roc_auc")
        diff = pivot["trained"] - pivot["fixed"]
        rows.append({"setting": "Trained vs fixed feature map (both quantum)",
                     "quantum_model": "trained",
                     "classical_model": "fixed quantum map", "quantum": pivot["trained"].mean(),
                     "classical": pivot["fixed"].mean(), "delta": diff.mean(),
                     "ci": 1.96 * diff.sem(), "seeds": len(diff)})
    return pd.DataFrame([r for r in rows if r])


def verdicts(scorecard: pd.DataFrame) -> pd.Series:
    """wins / ties / loses, by whether the interval clears zero."""
    lo, hi = scorecard["delta"] - scorecard["ci"], scorecard["delta"] + scorecard["ci"]
    return pd.Series(np.where(lo > 0, "quantum wins",
                              np.where(hi < 0, "classical wins", "tie")),
                     index=scorecard.index)
