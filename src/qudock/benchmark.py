"""Sample-efficiency benchmark: every model, several training sizes and seeds."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from qudock.classical import GRIDS, make_classical, scores
from qudock.data import load_bace, scaffold_split
from qudock.features import FeaturePipeline
from qudock.quantum import QSVC

log = logging.getLogger(__name__)

QSVC_GRID = {"C": [0.1, 1, 10], "bandwidth": [0.25, 0.5, 1.0]}


@dataclass
class BenchmarkConfig:
    n_qubits: int = 6
    sizes: list[int | None] = field(default_factory=lambda: [25, 50, 100, 200, 400, None])
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    models: list[str] = field(default_factory=lambda: [*GRIDS, "qsvc"])
    reps: int = 2
    test_frac: float = 0.2


def make_model(name: str, seed: int, n_train: int, reps: int = 2):
    if name != "qsvc":
        return make_classical(name, seed=seed, n_train=n_train)
    folds = 5 if n_train >= 100 else 3
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return GridSearchCV(QSVC(reps=reps, seed=seed), QSVC_GRID, cv=cv, scoring="roc_auc")


def metrics(y_true, score, y_pred) -> dict[str, float]:
    return {
        "roc_auc": roc_auc_score(y_true, score),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def subsample(y_train: np.ndarray, size: int | None, seed: int) -> np.ndarray:
    """Stratified subset of training indices (all of them when size is None)."""
    idx = np.arange(len(y_train))
    if size is None or size >= len(idx):
        return idx
    keep, _ = train_test_split(idx, train_size=size, stratify=y_train, random_state=seed)
    return np.sort(keep)


def run_benchmark(cfg: BenchmarkConfig, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return one row per (model, size, seed) with test-set metrics."""
    df = load_bace() if df is None else df
    split = scaffold_split(df["smiles"], test_frac=cfg.test_frac)
    smiles_tr, y_tr = df["smiles"].values[split.train], df["label"].values[split.train]
    smiles_te, y_te = df["smiles"].values[split.test], df["label"].values[split.test]

    rows = []
    for size in cfg.sizes:
        for seed in cfg.seeds:
            keep = subsample(y_tr, size, seed)
            # features are refit on each subset so small runs never see more data
            pipe = FeaturePipeline(cfg.n_qubits, seed=seed)
            X_tr = pipe.fit_transform(smiles_tr[keep])
            X_te = pipe.transform(smiles_te)
            for name in cfg.models:
                start = time.perf_counter()
                model = make_model(name, seed, len(keep), cfg.reps).fit(X_tr, y_tr[keep])
                row = {
                    "model": name, "n_train": len(keep), "seed": seed,
                    "n_qubits": cfg.n_qubits, "seconds": time.perf_counter() - start,
                    **metrics(y_te, scores(model, X_te), model.predict(X_te)),
                }
                log.info("%s n=%d seed=%d auc=%.3f", name, len(keep), seed, row["roc_auc"])
                rows.append(row)
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame, metric: str = "roc_auc") -> pd.DataFrame:
    """Mean and std of a metric per model and training size."""
    return (results.groupby(["n_train", "model"])[metric]
            .agg(["mean", "std"]).reset_index())


def compare_to_best_classical(results: pd.DataFrame, metric: str = "roc_auc") -> pd.DataFrame:
    """Paired Wilcoxon test of QSVC vs the best classical model at each size.

    Sizes where QSVC or every classical model is missing are skipped.
    """
    out = []
    for n, group in results.groupby("n_train"):
        wide = group.pivot(index="seed", columns="model", values=metric)
        if "qsvc" not in wide or wide.shape[1] < 2:
            continue
        classical = wide.drop(columns="qsvc")
        best = classical.mean().idxmax()
        diff = wide["qsvc"] - wide[best]
        p = wilcoxon(diff).pvalue if len(diff) >= 5 and diff.any() else float("nan")
        out.append({"n_train": n, "best_classical": best, "qsvc": wide["qsvc"].mean(),
                    "classical": wide[best].mean(), "delta": diff.mean(), "p_value": p})
    return pd.DataFrame(out, columns=["n_train", "best_classical", "qsvc", "classical",
                                      "delta", "p_value"])


def bootstrap_ci(y_true, score, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Bootstrap confidence interval for ROC-AUC on a fixed test set."""
    rng = np.random.default_rng(seed)
    y_true, score = np.asarray(y_true), np.asarray(score)
    aucs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[i])) == 2:
            aucs.append(roc_auc_score(y_true[i], score[i]))
    if not aucs:
        raise ValueError("every bootstrap resample had a single class; need both labels")
    return tuple(np.quantile(aucs, [alpha / 2, 1 - alpha / 2]))
