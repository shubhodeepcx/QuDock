"""Does the answer depend on the one split we chose?

Every number in this project so far comes from a single Bemis-Murcko scaffold
split. That is the standard way to do it, but it means one arbitrary choice of
which rare scaffolds to hold out sits underneath every claim, and a gap of 0.01
ROC-AUC is not worth much if it moves by 0.05 when the split changes.

So we repeat the whole comparison on five equally hard splits: crowded scaffolds
stay in training, but which rare ones are held out is reseeded, which changes
about a third of the test set each time. Each split gets its own best quantum
model and its own best classical model, both chosen by inner cross-validation on
training molecules only, and the number we report is the gap between them, split
by split. If the conclusion survives five splits it is a conclusion; if it does
not, the spread is the honest headline.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from qudock.final_study import FinalConfig, run_final_study
from qudock.scorecard import CLASSICAL, QUANTUM

log = logging.getLogger(__name__)

# every model the scorecard can pick as "best quantum" or "best classical", so the
# comparison on a reseeded split is the same comparison the scorecard makes
MODELS = ["qsvc_pca", "qsvc_pls", "hybrid_pca", "hybrid_pls", "projected_pls",
          "svm_rbf_pls", "rf_pls", "tanimoto_svm", "rf_fingerprint"]


def run_robustness_study(splits=(-1, 0, 1, 2, 3, 4), n_qubits: int = 6, seeds=(0, 1),
                         size: int | None = None, models=MODELS,
                         df: pd.DataFrame | None = None) -> pd.DataFrame:
    """The final-study models on several scaffold splits of the same dataset.

    Split -1 is the fixed split every other study in this project uses, so it can
    be read next to the reseeded ones on exactly the same code path.
    """
    rows = []
    for split_seed in splits:
        cfg = FinalConfig(n_qubits=n_qubits, sizes=[size], seeds=list(seeds),
                          models=list(models),
                          split_seed=None if split_seed == -1 else split_seed)
        runs, _ = run_final_study(cfg, df=df)
        runs["split_seed"] = split_seed
        rows.append(runs)
        log.info("split %d done, best auc %.3f", split_seed, runs["roc_auc"].max())
    return pd.concat(rows, ignore_index=True)


def summarise_robustness(results: pd.DataFrame) -> pd.DataFrame:
    """Per split: the best quantum model, the best classical model and the gap."""
    out = []
    for split_seed, g in results.groupby("split_seed"):
        means = g.groupby("model")["roc_auc"].mean()
        quantum = means[[m for m in means.index if m in QUANTUM]]
        classical = means[[m for m in means.index if m in CLASSICAL]]
        out.append({"split_seed": split_seed,
                    "best_quantum": quantum.idxmax(), "quantum_auc": quantum.max(),
                    "best_classical": classical.idxmax(), "classical_auc": classical.max(),
                    "delta": quantum.max() - classical.max()})
    return pd.DataFrame(out)


def per_model_spread(results: pd.DataFrame) -> pd.DataFrame:
    """Mean ROC-AUC of each model across splits, with the spread between them."""
    per_split = results.groupby(["model", "split_seed"])["roc_auc"].mean().reset_index()
    g = per_split.groupby("model")["roc_auc"]
    out = pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=1), "worst": g.min(),
                        "best": g.max()}).reset_index()
    out["range"] = out["best"] - out["worst"]
    return out.sort_values("mean", ascending=False).reset_index(drop=True)


def verdict(summary: pd.DataFrame) -> dict[str, float]:
    """One line a judge can read: how often, and by how much, the quantum side wins."""
    delta = summary["delta"].to_numpy()
    return {"n_splits": len(delta), "mean_delta": float(delta.mean()),
            "worst_delta": float(delta.min()), "best_delta": float(delta.max()),
            "splits_quantum_ahead": int((delta > 0).sum()),
            "spread": float(delta.max() - delta.min()),
            # the gap moves more between splits than it does between the models
            "bigger_than_gap": bool(abs(delta.max() - delta.min()) > abs(delta.mean())
                                    if len(delta) else False)}


def best_split_free_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Every model's mean over splits and seeds, which is the number worth quoting."""
    metrics = [c for c in ("roc_auc", "pr_auc", "ef_1", "ef_5", "bedroc") if c in results]
    out = results.groupby("model")[metrics].mean().reset_index()
    per_split = results.groupby(["model", "split_seed"])["roc_auc"].mean()
    # the interval that matters is the spread between splits, not between seeds
    out["roc_auc_ci"] = out["model"].map(1.96 * per_split.groupby("model").sem().fillna(0))
    return out.sort_values("roc_auc", ascending=False).reset_index(drop=True)


def paired_split_test(results: pd.DataFrame, quantum: str, classical: str) -> dict[str, float]:
    """Compare two named models split by split, which is the paired test that fits here."""
    wide = results.groupby(["split_seed", "model"])["roc_auc"].mean().unstack()
    if quantum not in wide or classical not in wide:
        return {}
    diff = (wide[quantum] - wide[classical]).to_numpy()
    n = len(diff)
    se = diff.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return {"model": quantum, "versus": classical, "n_splits": n,
            "mean_difference": float(diff.mean()),
            "ci_low": float(diff.mean() - 1.96 * se), "ci_high": float(diff.mean() + 1.96 * se),
            "splits_ahead": int((diff > 0).sum())}


def split_versus_seed_noise(results: pd.DataFrame) -> pd.DataFrame:
    """How much of a model's wobble is the split, and how much is the seed.

    Between splits: the spread of a model's per-split mean. Within a split: the
    spread between seeds, averaged over splits. A difference that is smaller than
    either number is not a finding, whichever way it points.
    """
    per_split = results.groupby(["model", "split_seed"])["roc_auc"].mean()
    between = per_split.groupby("model").std(ddof=1)
    within = results.groupby(["model", "split_seed"])["roc_auc"].std(ddof=1) \
        .groupby("model").mean()
    out = pd.DataFrame({"between_splits": between, "within_split_between_seeds": within})
    return out.fillna(0).reset_index()
