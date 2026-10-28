"""QuDock-Bench: one fixed protocol for scoring any molecule classifier, quantum or not.

Every model is scored on the same tasks, the same Bemis-Murcko scaffold splits,
the same training subsamples and the same seeds, with the same screening
metrics, so numbers from different people can sit in one table.

A model is anything that turns SMILES into scores. Write a function that takes
a :class:`TaskInfo` and returns an object with ``fit(smiles, labels)`` and
``decision_function(smiles)`` (or ``predict_proba``), then pass it as
``module.path:function`` to ``qudock bench --models``. :class:`SmilesModel`
wraps any featuriser plus scikit-learn estimator into that shape.

Tasks (MoleculeNet, cleaned the same way as the rest of QuDock):

* ``bace``: BACE-1 inhibition, the Alzheimer's target this project is about
* ``bbbp``: blood-brain barrier penetration, which any Alzheimer's drug needs
* ``clintox``: failed clinical trials for toxicity (heavily imbalanced, 6% positive)
"""

from __future__ import annotations

import importlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from qudock.benchmark import subsample
from qudock.classical import scores
from qudock.data import DATASETS, scaffold_split
from qudock.features import FeaturePipeline, descriptors, morgan_fingerprints
from qudock.screening import screening_metrics

log = logging.getLogger(__name__)

TASKS = ("bace", "bbbp", "clintox")


@dataclass
class TaskInfo:
    """What a model factory is told about the run it is built for."""
    task: str
    n_train: int
    seed: int


def _cv(info: TaskInfo, y=None) -> StratifiedKFold:
    folds = 5 if info.n_train >= 100 else 3
    if y is not None:                           # rare positives: every fold needs a few
        folds = max(2, min(folds, int(np.bincount(np.asarray(y)).min())))
    return StratifiedKFold(folds, shuffle=True, random_state=info.seed)


class SmilesModel:
    """featurise(smiles) -> array, then a scikit-learn estimator, tuned by inner CV if a grid
    is given."""

    def __init__(self, featurise: Callable, estimator, grid: dict | None = None,
                 info: TaskInfo | None = None, needs_labels: bool = False):
        self.featurise = featurise
        self.estimator = estimator
        self.grid = grid
        self.info = info
        self.needs_labels = needs_labels        # featuriser is fit on training labels

    def fit(self, smiles, y):
        y = np.asarray(y)
        X = self.featurise(list(smiles), y) if self.needs_labels else self.featurise(list(smiles))
        model = clone(self.estimator)
        if self.grid:
            model = GridSearchCV(model, self.grid, cv=_cv(self.info, y), scoring="roc_auc")
        self.model_ = model.fit(X, y)
        return self

    def decision_function(self, smiles):
        X = self.featurise(list(smiles), None) if self.needs_labels else self.featurise(
            list(smiles))
        return scores(self.model_, X)


def _fp_desc(smiles):
    return np.hstack([morgan_fingerprints(smiles), descriptors(smiles)])


class _PLSAngles:
    """PLS compression to qubit angles that is fit once on the training labels."""

    def __init__(self, n_qubits: int, seed: int):
        self.pipe = FeaturePipeline(n_qubits, seed=seed, method="pls", use_descriptors=True)

    def __call__(self, smiles, y):
        if y is not None:
            self.pipe.fit(smiles, y)
        return self.pipe.transform(smiles)


class _PLSPlusFingerprint(_PLSAngles):
    def __call__(self, smiles, y):
        from qudock.kernels import stack

        return stack(super().__call__(smiles, y), morgan_fingerprints(smiles))


# ---------------------------------------------------------------- built-in entries

def rf_fingerprint(info: TaskInfo):
    rf = RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=info.seed,
                                class_weight="balanced")
    return SmilesModel(morgan_fingerprints, rf, {"min_samples_leaf": [1, 3]}, info)


def gb_fingerprint(info: TaskInfo):
    gb = HistGradientBoostingClassifier(max_iter=300, random_state=info.seed,
                                        class_weight="balanced")
    return SmilesModel(_fp_desc, gb, {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31]},
                       info)


def tanimoto_svm(info: TaskInfo):
    from qudock.kernels import KernelSVM

    return SmilesModel(_PLSPlusFingerprint(6, info.seed),
                       KernelSVM(kind="tanimoto", n_angles=6, seed=info.seed),
                       {"C": [0.1, 1, 10]}, info, needs_labels=True)


def qsvc_pls(info: TaskInfo):
    from qudock.kernels import KernelSVM

    return SmilesModel(_PLSAngles(6, info.seed),
                       KernelSVM(kind="quantum", n_angles=6, seed=info.seed),
                       {"C": [0.1, 1, 10], "bandwidth": [0.25, 0.5, 1.0]}, info,
                       needs_labels=True)


def hybrid_pls(info: TaskInfo):
    from qudock.kernels import KernelSVM

    return SmilesModel(_PLSPlusFingerprint(6, info.seed),
                       KernelSVM(kind="hybrid", n_angles=6, bandwidth=0.5, seed=info.seed),
                       {"C": [0.1, 1, 10], "weight": [0.1, 0.25, 0.5]}, info, needs_labels=True)


def nystroem_projected(info: TaskInfo):
    from qudock.kernels import NystroemSVM

    return SmilesModel(_PLSAngles(6, info.seed),
                       NystroemSVM(kind="projected", n_landmarks=150, seed=info.seed),
                       {"C": [0.1, 1], "gamma": [0.2, 1.0]}, info, needs_labels=True)


def graph_quantum(info: TaskInfo):
    from sklearn.svm import SVC

    from qudock.graphmap import GraphKernel, graph_angles

    class GraphSVM(SVC):
        def __init__(self, C=1.0, bandwidth=0.5, random_state=None):
            super().__init__(C=C, kernel=self._kernel, random_state=random_state)
            self.bandwidth = bandwidth

        def _kernel(self, A, B):
            return GraphKernel(bandwidth=self.bandwidth)(A, B)

    return SmilesModel(graph_angles, GraphSVM(random_state=info.seed),
                       {"C": [0.1, 1, 10], "bandwidth": [0.25, 0.5, 1.0]}, info)


BUILTIN = {f.__name__: f for f in (rf_fingerprint, gb_fingerprint, tanimoto_svm, qsvc_pls,
                                   hybrid_pls, nystroem_projected, graph_quantum)}
QUANTUM = {"qsvc_pls", "hybrid_pls", "nystroem_projected", "graph_quantum"}


def resolve(spec: str) -> Callable[[TaskInfo], object]:
    """A built-in name, or ``package.module:function`` for your own model factory."""
    if spec in BUILTIN:
        return BUILTIN[spec]
    if ":" not in spec:
        raise ValueError(f"unknown model {spec!r}: use one of {sorted(BUILTIN)} "
                         "or module.path:function")
    module, name = spec.split(":", 1)
    return getattr(importlib.import_module(module), name)


def run_bench(models, tasks=TASKS, sizes=(None,), seeds=(0, 1, 2),
              datasets: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """One row per (task, model, size, seed) with screening metrics on the scaffold test set."""
    rows = []
    for task in tasks:
        df = (datasets or {}).get(task)
        df = DATASETS[task]() if df is None else df
        split = scaffold_split(df["smiles"])
        smiles, y = df["smiles"].values, df["label"].values
        smi_tr, y_tr, smi_te, y_te = (smiles[split.train], y[split.train],
                                      smiles[split.test], y[split.test])
        for size in sizes:
            for seed in seeds:
                keep = subsample(y_tr, size, seed)
                for spec in models:
                    info = TaskInfo(task, len(keep), seed)
                    start = time.perf_counter()
                    model = resolve(spec)(info).fit(smi_tr[keep], y_tr[keep])
                    score = np.asarray(model.decision_function(smi_te), dtype=float)
                    rows.append({"task": task, "model": spec, "n_train": len(keep),
                                 "seed": seed, "seconds": time.perf_counter() - start,
                                 **screening_metrics(y_te, score)})
                    log.info("%s %s n=%d seed=%d auc=%.3f", task, spec, len(keep), seed,
                             rows[-1]["roc_auc"])
    return pd.DataFrame(rows)


def leaderboard(results: pd.DataFrame, metric: str = "roc_auc") -> pd.DataFrame:
    """Mean metric and 95% interval per task at the largest training size, plus mean rank."""
    largest = results.groupby("task")["n_train"].transform("max")
    full = results[results["n_train"] == largest]
    g = full.groupby(["model", "task"])[metric]
    table = (g.mean().unstack("task").round(3).astype(str) + " ± "
             + (1.96 * g.sem().fillna(0)).unstack("task").round(3).astype(str))
    ranks = g.mean().unstack("task").rank(ascending=False).mean(axis=1)
    table.insert(0, "mean_rank", ranks.round(2))
    table.insert(1, "quantum", [m in QUANTUM for m in table.index])
    return table.sort_values("mean_rank")
