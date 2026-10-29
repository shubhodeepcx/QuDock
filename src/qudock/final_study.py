"""Final-round study: can a better-fed or hybrid quantum kernel close the gap?

Compared on the same scaffold split, training sizes and seeds:

* baselines from the first round on PCA angles: SVM-RBF, Random Forest, QSVC
* QSVC on PLS angles (supervised compression), bandwidth tuned by alignment,
  next to SVM-RBF and Random Forest on the same PLS angles
* projected quantum kernel on PLS angles
* Tanimoto SVM and Random Forest on the full fingerprint (the classical ceiling)
* gradient boosting on the fingerprint plus RDKit descriptors, the strongest
  off-the-shelf tabular baseline
* hybrid: weight * quantum + (1 - weight) * Tanimoto, weight chosen by inner CV,
  with the quantum part on PCA angles (unsupervised) or PLS angles

PLS is fit on all training labels, so inner CV of PLS-based models is
slightly optimistic; the PCA hybrid has no such leak.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from qudock.benchmark import subsample
from qudock.classical import make_classical, scores
from qudock.data import load_bace, scaffold_split
from qudock.features import FeaturePipeline, descriptors, morgan_fingerprints
from qudock.kernels import KernelSVM, stack, tune_bandwidth
from qudock.screening import paired_bootstrap, screening_metrics

log = logging.getLogger(__name__)

MODELS = ["svm_rbf_pca", "rf_pca", "qsvc_pca", "svm_rbf_pls", "rf_pls", "qsvc_pls",
          "projected_pls", "tanimoto_svm", "rf_fingerprint", "gb_fingerprint", "hybrid_pca",
          "hybrid_pls"]


@dataclass
class FinalConfig:
    n_qubits: int = 6
    sizes: list[int | None] = field(default_factory=lambda: [50, 100, 200, 400, None])
    seeds: list[int] = field(default_factory=lambda: list(range(10)))
    models: list[str] = field(default_factory=lambda: list(MODELS))
    reps: int = 2
    test_frac: float = 0.2
    split_seed: int | None = None       # None: the one fixed scaffold split used everywhere


def _cv(seed: int, n: int):
    return StratifiedKFold(n_splits=5 if n >= 100 else 3, shuffle=True, random_state=seed)


def build(name: str, seed: int, n: int, n_qubits: int, bw_pca: float, bw_pls: float, reps: int):
    """Estimator for one model; inputs are chosen in ``inputs``."""
    if name in ("svm_rbf_pca", "svm_rbf_pls"):
        return make_classical("svm_rbf", seed=seed, n_train=n)
    if name in ("rf_pca", "rf_pls"):
        return make_classical("random_forest", seed=seed, n_train=n)
    if name == "rf_fingerprint":
        rf = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1)
        return GridSearchCV(rf, {"min_samples_leaf": [1, 3]}, cv=_cv(seed, n), scoring="roc_auc")
    if name == "gb_fingerprint":
        gb = HistGradientBoostingClassifier(max_iter=300, random_state=seed)
        grid = {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31],
                "l2_regularization": [0.0, 1.0]}
        return GridSearchCV(gb, grid, cv=_cv(seed, n), scoring="roc_auc")
    base = {"n_angles": n_qubits, "reps": reps, "seed": seed}
    grids = {
        "qsvc_pca": ("quantum", {"C": [0.1, 1, 10], "bandwidth": [bw_pca]}),
        "qsvc_pls": ("quantum", {"C": [0.1, 1, 10], "bandwidth": [bw_pls]}),
        "projected_pls": ("projected", {"C": [0.1, 1, 10], "bandwidth": [0.5, 1.0],
                                        "gamma": [0.05, 0.2, 1.0, 5.0]}),
        "tanimoto_svm": ("tanimoto", {"C": [0.1, 1, 10]}),
        "hybrid_pca": ("hybrid", {"C": [0.1, 1, 10], "bandwidth": [bw_pca],
                                  "weight": [0.0, 0.1, 0.25, 0.5]}),
        "hybrid_pls": ("hybrid", {"C": [0.1, 1, 10], "bandwidth": [bw_pls],
                                  "weight": [0.0, 0.1, 0.25, 0.5]}),
    }
    kind, grid = grids[name]
    return GridSearchCV(KernelSVM(kind=kind, **base), grid, cv=_cv(seed, n), scoring="roc_auc")


def inputs(name: str, pca, pls, fp, desc=None):
    """(train, test) inputs for a model from the PCA, PLS, fingerprint and descriptor features."""
    if name in ("svm_rbf_pca", "rf_pca"):
        return pca
    if name in ("svm_rbf_pls", "rf_pls"):
        return pls
    if name == "rf_fingerprint":
        return fp
    if name == "gb_fingerprint":
        return np.hstack([fp[0], desc[0]]), np.hstack([fp[1], desc[1]])
    angles = pca if name in ("qsvc_pca", "hybrid_pca") else pls
    return stack(angles[0], fp[0]), stack(angles[1], fp[1])


def run_final_study(cfg: FinalConfig, df: pd.DataFrame | None = None):
    """Return per-run metrics, and test-set scores of every model at full size (seed 0)."""
    df = load_bace() if df is None else df
    split = scaffold_split(df["smiles"], test_frac=cfg.test_frac, seed=cfg.split_seed)
    smi_tr, y_tr = df["smiles"].values[split.train], df["label"].values[split.train]
    smi_te, y_te = df["smiles"].values[split.test], df["label"].values[split.test]
    fp_all_tr, fp_te = morgan_fingerprints(smi_tr), morgan_fingerprints(smi_te)
    desc_all_tr, desc_te = descriptors(smi_tr), descriptors(smi_te)

    rows, full_scores = [], {}
    for size in cfg.sizes:
        for seed in cfg.seeds:
            keep = subsample(y_tr, size, seed)
            n = len(keep)
            pca_pipe = FeaturePipeline(cfg.n_qubits, seed=seed)
            pca = pca_pipe.fit_transform(smi_tr[keep]), pca_pipe.transform(smi_te)
            pls_pipe = FeaturePipeline(cfg.n_qubits, seed=seed, method="pls",
                                       use_descriptors=True)
            pls = pls_pipe.fit_transform(smi_tr[keep], y_tr[keep]), pls_pipe.transform(smi_te)
            fp = fp_all_tr[keep], fp_te
            desc = desc_all_tr[keep], desc_te
            bw_pca = tune_bandwidth(pca[0], y_tr[keep], cfg.n_qubits, cfg.reps, seed=seed)
            bw_pls = tune_bandwidth(pls[0], y_tr[keep], cfg.n_qubits, cfg.reps, seed=seed)
            for name in cfg.models:
                start = time.perf_counter()
                X_tr, X_te = inputs(name, pca, pls, fp, desc)
                model = build(name, seed, n, cfg.n_qubits, bw_pca, bw_pls, cfg.reps)
                model.fit(X_tr, y_tr[keep])
                s = scores(model, X_te)
                params = getattr(model, "best_params_", {})
                rows.append({"model": name, "n_train": n, "seed": seed,
                             "n_qubits": cfg.n_qubits, "bandwidth_pca": bw_pca,
                             "bandwidth_pls": bw_pls, "weight": params.get("weight"),
                             "seconds": time.perf_counter() - start,
                             **screening_metrics(y_te, s)})
                log.info("%s n=%d seed=%d auc=%.3f", name, n, seed, rows[-1]["roc_auc"])
                if size is None and seed == cfg.seeds[0]:
                    full_scores[name] = s
    scores_df = pd.DataFrame({"label": y_te, **full_scores}) if full_scores else None
    return pd.DataFrame(rows), scores_df


def summarise_final(results: pd.DataFrame) -> pd.DataFrame:
    """Mean and 95% interval over seeds of each metric, per model and size."""
    metrics = ["roc_auc", "pr_auc", "ef_1", "ef_5", "bedroc"]
    g = results.groupby(["n_train", "model"])[metrics]
    mean, sem = g.mean(), g.sem().fillna(0)
    out = mean.copy()
    out["roc_auc_ci"] = 1.96 * sem["roc_auc"]
    return out.reset_index()


def compare_pairs(scores_df: pd.DataFrame, pairs) -> pd.DataFrame:
    """Paired bootstrap of ROC-AUC differences on the full-size test set."""
    y = scores_df["label"].values
    return pd.DataFrame([{"model": a, "versus": b,
                          **paired_bootstrap(y, scores_df[a].values, scores_df[b].values)}
                         for a, b in pairs if a in scores_df and b in scores_df])
