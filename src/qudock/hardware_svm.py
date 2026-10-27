"""Train and test a quantum-kernel SVM on a real IBM Quantum device.

The first-round hardware run only checked a 12-molecule kernel. Here the
projected quantum kernel makes a full classifier affordable: each molecule
needs one circuit (its single-qubit X, Y and Z expectations), not one per
pair, so a few hundred molecules fit in a couple of minutes of quantum time.

The same features are computed exactly, so the device classifier can be
compared with the ideal one on the same molecules. Device features are saved,
so the analysis can be redone later without spending more quantum time.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit.circuit.library import zz_feature_map
from qiskit.transpiler import generate_preset_pass_manager
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

from qudock.data import load_bace, scaffold_split
from qudock.features import FeaturePipeline
from qudock.hardware import pick_molecules, select_backend
from qudock.kernels import ProjectedKernel, _pauli_ops
from qudock.screening import screening_metrics

log = logging.getLogger(__name__)


@dataclass
class HardwareSVMConfig:
    n_qubits: int = 4
    n_train: int = 120
    n_test: int = 60
    reps: int = 2
    bandwidth: float = 1.0
    precision: float = 0.03            # target standard error per expectation value
    backend: str | None = None         # None = least busy device
    max_execution_time: int = 300      # hard cap on quantum seconds billed
    seed: int = 0


def prepare(cfg: HardwareSVMConfig, df: pd.DataFrame | None = None):
    """PLS angles and labels for balanced train and test molecules from the scaffold split."""
    df = load_bace() if df is None else df
    split = scaffold_split(df["smiles"])
    smiles, y = df["smiles"].values, df["label"].values
    pipe = FeaturePipeline(cfg.n_qubits, seed=cfg.seed, method="pls", use_descriptors=True)
    pipe.fit(smiles[split.train], y[split.train])
    tr = split.train[pick_molecules(y[split.train], cfg.n_train, cfg.seed)]
    te = split.test[pick_molecules(y[split.test], cfg.n_test, cfg.seed)]
    return (pipe.transform(smiles[tr]), y[tr]), (pipe.transform(smiles[te]), y[te])


def exact_features(X: np.ndarray, cfg: HardwareSVMConfig) -> np.ndarray:
    return ProjectedKernel(cfg.n_qubits, cfg.reps, cfg.bandwidth).features(X)


def measure_features(X: np.ndarray, cfg: HardwareSVMConfig, backend, mitigate: bool = True):
    """Single-qubit Pauli expectations of every molecule's state, from one Estimator job."""
    from qiskit_ibm_runtime import EstimatorV2

    circuit = zz_feature_map(cfg.n_qubits, reps=cfg.reps, entanglement="linear")
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend,
                                      seed_transpiler=cfg.seed)
    isa = pm.run(circuit)
    # shape (3 * n_qubits, 1) broadcasts against (n_molecules, n_params) parameter values
    observables = [[op.apply_layout(isa.layout)] for op in _pauli_ops(cfg.n_qubits)]
    estimator = EstimatorV2(mode=backend)
    estimator.options.default_precision = cfg.precision
    if mitigate:
        estimator.options.resilience_level = 1          # readout error mitigation (TREX)
        estimator.options.dynamical_decoupling.enable = True
        estimator.options.twirling.enable_gates = True
        estimator.options.max_execution_time = cfg.max_execution_time
    params = cfg.bandwidth * np.asarray(X)
    t0 = time.time()
    job = estimator.run([(isa, observables, params)])
    log.info("submitted job %s to %s", job.job_id(), backend.name)
    evs = np.asarray(job.result()[0].data.evs)          # (3 * n_qubits, n_molecules)
    usage = job.usage() if hasattr(job, "usage") else None
    info = {"backend": backend.name, "job_id": job.job_id(), "n_molecules": len(X),
            "n_observables": len(observables), "precision": cfg.precision,
            "two_qubit_depth": isa.depth(lambda g: g.operation.num_qubits == 2),
            "quantum_seconds": usage, "wall_seconds": round(time.time() - t0, 1),
            "mitigation": mitigate, "date": time.strftime("%Y-%m-%d %H:%M %Z")}
    return evs.T, info


def fit_projected_svm(F_train, y_train, n_qubits: int, seed: int = 0):
    """SVM on a projected kernel built from precomputed Pauli features; gamma and C by CV."""
    pk = ProjectedKernel(n_qubits)
    best = None
    for gamma in (0.05, 0.2, 1.0, 5.0):
        pk.gamma = gamma
        K = pk.from_features(F_train)
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        search = GridSearchCV(SVC(kernel="precomputed", random_state=seed),
                              {"C": [0.1, 1, 10]}, cv=cv, scoring="roc_auc").fit(K, y_train)
        if best is None or search.best_score_ > best[0]:
            best = (search.best_score_, gamma, search.best_estimator_)
    _, gamma, svc = best
    return svc, gamma


def evaluate(F_train, y_train, F_test, y_test, n_qubits: int, seed: int = 0) -> dict:
    svc, gamma = fit_projected_svm(F_train, y_train, n_qubits, seed)
    pk = ProjectedKernel(n_qubits, gamma=gamma)
    score = svc.decision_function(pk.from_features(F_test, F_train))
    return {"gamma": gamma, "C": svc.C, **screening_metrics(y_test, score)}


def run_hardware_svm(cfg: HardwareSVMConfig, backend=None, df: pd.DataFrame | None = None):
    """Device vs exact projected-kernel SVM on the same molecules."""
    (X_tr, y_tr), (X_te, y_te) = prepare(cfg, df)
    backend = backend or select_backend(cfg.backend)
    X_all = np.vstack([X_tr, X_te])
    F_dev, info = measure_features(X_all, cfg, backend,
                                   mitigate=not backend.name.startswith("fake"))
    F_exact = exact_features(X_all, cfg)
    n = len(X_tr)
    saved = {"angles": X_all, "labels": np.r_[y_tr, y_te], "n_train": n,
             "exact_features": F_exact, "device_features": F_dev}
    info["config"] = asdict(cfg)
    return analyse(saved, cfg), saved, info


def analyse(saved: dict, cfg: HardwareSVMConfig) -> pd.DataFrame:
    n, y = int(saved["n_train"]), saved["labels"]
    rows = []
    for label, F in [("exact simulation", saved["exact_features"]),
                     ("device", saved["device_features"])]:
        rows.append({"features": label,
                     "feature_error": float(np.abs(F - saved["exact_features"]).mean()),
                     **evaluate(F[:n], y[:n], F[n:], y[n:], cfg.n_qubits, cfg.seed)})
    # the classifier trained on exact features, applied to device features of new molecules
    svc, gamma = fit_projected_svm(saved["exact_features"][:n], y[:n], cfg.n_qubits, cfg.seed)
    pk = ProjectedKernel(cfg.n_qubits, gamma=gamma)
    score = svc.decision_function(pk.from_features(saved["device_features"][n:],
                                                   saved["exact_features"][:n]))
    rows.append({"features": "trained exact, tested on device", "feature_error": float("nan"),
                 "gamma": gamma, "C": svc.C, **screening_metrics(y[n:], score)})
    return pd.DataFrame(rows)


def save(rows: pd.DataFrame, saved: dict, info: dict, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / "hardware_svm.csv", index=False)
    np.savez(out / "hardware_svm_features.npz", **saved)
    (out / "hardware_svm_run.json").write_text(json.dumps(info, indent=2, default=str) + "\n")


def reanalyse(out: Path, cfg: HardwareSVMConfig | None = None) -> pd.DataFrame:
    """Redo the analysis from saved device features, no device needed."""
    saved = dict(np.load(out / "hardware_svm_features.npz"))
    return analyse(saved, cfg or HardwareSVMConfig())
