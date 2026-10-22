"""Noise study: QSVC with kernels estimated from shots, with and without device noise.

Small on purpose (about a hundred molecules, 4 qubits), matching what a real
hardware run could afford.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

from qudock.benchmark import subsample
from qudock.data import load_bace, scaffold_split
from qudock.features import FeaturePipeline
from qudock.noise import SampledZZKernel, kernel_error, make_simulator, nearest_psd
from qudock.quantum import ZZKernel

log = logging.getLogger(__name__)


@dataclass
class NoiseConfig:
    n_qubits: int = 4
    n_train: int = 80
    n_test: int = 40
    bandwidth: float = 0.5
    C: float = 1.0
    shots: list[int] = field(default_factory=lambda: [256, 1024, 4096])
    reps: list[int] = field(default_factory=lambda: [1, 2])
    backend: str = "FakeTorino"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])


def fake_backend(name: str):
    from qiskit_ibm_runtime import fake_provider
    return getattr(fake_provider, name)()


def _auc(K_train, K_test, y_train, y_test, C):
    svc = SVC(kernel="precomputed", C=C).fit(K_train, y_train)
    return roc_auc_score(y_test, svc.decision_function(K_test))


def run_noise_study(cfg: NoiseConfig, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (seed, reps, noise setting, shots) with kernel error and test ROC-AUC."""
    df = load_bace() if df is None else df
    split = scaffold_split(df["smiles"])
    smiles, y = df["smiles"].values, df["label"].values
    backend = fake_backend(cfg.backend) if cfg.backend else None

    rows = []
    for seed in cfg.seeds:
        tr = split.train[subsample(y[split.train], cfg.n_train, seed)]
        te = split.test[subsample(y[split.test], cfg.n_test, seed)]
        pipe = FeaturePipeline(cfg.n_qubits, seed=seed)
        X_tr, X_te = pipe.fit_transform(smiles[tr]), pipe.transform(smiles[te])

        for reps in cfg.reps:
            exact = ZZKernel(cfg.n_qubits, reps=reps, bandwidth=cfg.bandwidth)
            K_tr, K_te = exact(X_tr), exact(X_te, X_tr)
            base = {"seed": seed, "reps": reps}
            rows.append({**base, "noise": "exact", "shots": np.inf, "psd_repair": False,
                         "rel_frobenius": 0.0, "alignment_to_exact": 1.0,
                         "roc_auc": _auc(K_tr, K_te, y[tr], y[te], cfg.C)})

            for noise in ["shots only", cfg.backend] if backend else ["shots only"]:
                sim = make_simulator(backend if noise != "shots only" else None, seed=seed)
                for shots in cfg.shots:
                    kernel = SampledZZKernel(cfg.n_qubits, reps=reps, bandwidth=cfg.bandwidth,
                                             shots=shots, simulator=sim, seed=seed)
                    N_tr, N_te = kernel(X_tr), kernel(X_te, X_tr)
                    for repair in (False, True):
                        Ktr = nearest_psd(N_tr) if repair else N_tr
                        rows.append({**base, "noise": noise, "shots": shots,
                                     "psd_repair": repair, **kernel_error(Ktr, K_tr),
                                     "roc_auc": _auc(Ktr, N_te, y[tr], y[te], cfg.C)})
                    log.info("seed=%d reps=%d %s shots=%d auc=%.3f", seed, reps, noise,
                             shots, rows[-2]["roc_auc"])
    return pd.DataFrame(rows)
