"""Does the quantum kernel survive more qubits?

As qubits are added, states of different molecules become almost orthogonal,
so every off-diagonal kernel value shrinks towards zero and the SVM sees no
structure left. This is kernel concentration, and it is the reason a quantum
kernel usually gets worse, not better, with width.

Here the same molecules are run at 4 to 14 qubits in three ways: a fixed
bandwidth, a bandwidth tuned per width by kernel-target alignment, and the
projected kernel, which reads single-qubit expectations instead of state
overlaps. The off-diagonal spread of each kernel is reported next to test
ROC-AUC, so the fix can be seen working rather than asserted.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from qudock.data import load_bace, scaffold_split
from qudock.features import FeaturePipeline, morgan_fingerprints
from qudock.kernels import KernelSVM, ProjectedKernel, stack, tune_bandwidth
from qudock.quantum import _STATE_CACHE, ZZKernel
from qudock.screening import screening_metrics

log = logging.getLogger(__name__)

KINDS = ["fixed bandwidth", "tuned bandwidth", "projected kernel"]


def off_diagonal_spread(K: np.ndarray) -> float:
    """Standard deviation of the off-diagonal kernel values; it falls as qubits grow."""
    off = ~np.eye(len(K), dtype=bool)
    return float(K[off].std())


def run_scaling_study(qubits=(4, 6, 8, 10, 12, 14), n_train: int = 400, seed: int = 0,
                      reps: int = 2, df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = load_bace() if df is None else df
    split = scaffold_split(df["smiles"])
    smiles, y = df["smiles"].values, df["label"].values
    rng = np.random.default_rng(seed)
    keep = rng.choice(split.train, size=min(n_train, len(split.train)), replace=False)
    smi_tr, y_tr, smi_te, y_te = smiles[keep], y[keep], smiles[split.test], y[split.test]
    fp_tr, fp_te = morgan_fingerprints(smi_tr), morgan_fingerprints(smi_te)

    rows = []
    for n_qubits in qubits:
        pipe = FeaturePipeline(n_qubits, seed=seed, method="pls", use_descriptors=True)
        A_tr, A_te = pipe.fit_transform(smi_tr, y_tr), pipe.transform(smi_te)
        tuned = tune_bandwidth(A_tr, y_tr, n_qubits, reps, seed=seed)
        X_tr, X_te = stack(A_tr, fp_tr), stack(A_te, fp_te)
        cv = StratifiedKFold(3, shuffle=True, random_state=seed)
        for kind in KINDS:
            start = time.perf_counter()
            bandwidth = tuned if kind == "tuned bandwidth" else 1.0
            grid = {"C": [0.1, 1, 10]}
            if kind == "projected kernel":
                base = KernelSVM(kind="projected", n_angles=n_qubits, reps=reps, seed=seed)
                grid["gamma"] = [0.05, 0.2, 1.0, 5.0]
                spread = off_diagonal_spread(ProjectedKernel(n_qubits, reps)(A_tr))
            else:
                base = KernelSVM(kind="quantum", n_angles=n_qubits, reps=reps,
                                 bandwidth=bandwidth, seed=seed)
                spread = off_diagonal_spread(ZZKernel(n_qubits, reps, bandwidth=bandwidth)(A_tr))
            model = GridSearchCV(base, grid, cv=cv, scoring="roc_auc").fit(X_tr, y_tr)
            rows.append({"n_qubits": n_qubits, "kind": kind, "bandwidth": bandwidth,
                         "off_diagonal_spread": spread,
                         "seconds": time.perf_counter() - start,
                         **screening_metrics(y_te, model.decision_function(X_te))})
            log.info("%d qubits, %s: spread %.4f auc %.3f", n_qubits, kind, spread,
                     rows[-1]["roc_auc"])
        _STATE_CACHE.clear()        # statevectors of this width are never reused
    return pd.DataFrame(rows)
