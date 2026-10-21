"""Qubit sweep: separate "fewer features" effects from "quantum" effects.

For each qubit count we record test ROC-AUC for every model and the kernel
diagnostics from ``qudock.analysis`` on a fixed training subsample.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel

from qudock.analysis import (
    effective_dimension,
    geometric_difference,
    kernel_target_alignment,
    off_diagonal_stats,
)
from qudock.benchmark import BenchmarkConfig, run_benchmark
from qudock.data import load_bace, scaffold_split
from qudock.features import FeaturePipeline
from qudock.quantum import ZZKernel

log = logging.getLogger(__name__)


def kernel_diagnostics(X: np.ndarray, y: np.ndarray, bandwidths=(0.25, 0.5, 1.0),
                       reps: int = 2) -> pd.DataFrame:
    """Geometry of the ZZ kernel against an RBF kernel on the same features."""
    n_qubits = X.shape[1]
    K_c = rbf_kernel(X, gamma=1 / (n_qubits * X.var()))
    rows = []
    for bw in bandwidths:
        K_q = ZZKernel(n_qubits, reps=reps, bandwidth=bw)(X)
        off = off_diagonal_stats(K_q)
        rows.append({
            "n_qubits": n_qubits, "bandwidth": bw,
            "geometric_difference": geometric_difference(K_c, K_q),
            "alignment_quantum": kernel_target_alignment(K_q, y),
            "alignment_rbf": kernel_target_alignment(K_c, y),
            "effective_dimension": effective_dimension(K_q),
            "offdiag_mean": off["mean"], "offdiag_var": off["var"],
        })
    return pd.DataFrame(rows)


def qubit_sweep(qubits=range(2, 11), seeds=(0, 1, 2), n_kernel: int = 300,
                df: pd.DataFrame | None = None):
    """Return (performance, diagnostics) frames across qubit counts."""
    df = load_bace() if df is None else df
    split = scaffold_split(df["smiles"])
    rng = np.random.default_rng(0)
    sample = rng.choice(split.train, size=min(n_kernel, len(split.train)), replace=False)

    perf, diag = [], []
    for n in qubits:
        log.info("qubits=%d", n)
        perf.append(run_benchmark(BenchmarkConfig(n_qubits=n, sizes=[None], seeds=list(seeds)),
                                  df=df))
        X = FeaturePipeline(n).fit_transform(df["smiles"].values[sample])
        diag.append(kernel_diagnostics(X, df["label"].values[sample]))
    return pd.concat(perf, ignore_index=True), pd.concat(diag, ignore_index=True)
