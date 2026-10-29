"""What the quantum kernel costs, in seconds and in circuits.

Accuracy tables leave out the other half of the question a reviewer should ask:
what does it take to get that number, and what would it take on a real device?

Two costs are reported. The first is measured here: wall-clock seconds to train
a model and score the test set, on one laptop CPU core, simulated. The second is
counted, not guessed: a fidelity kernel needs one circuit per pair of molecules,
so a training set of n needs n(n+1)/2 circuits plus n * m more to score m test
molecules, while a projected kernel needs one circuit per molecule, n + m in
total. Classical models need none.

Those circuit counts are turned into device time at the rate we actually
measured on ibm_marrakesh: 66 circuits at 1,024 shots in 20 seconds of quantum
time, which is 0.30 seconds per circuit. That is the honest way to answer "could
you run this for real": not with an opinion, but with our own job's throughput
multiplied by the number of circuits the method needs.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from qudock.data import load_bace, scaffold_split
from qudock.features import FeaturePipeline, morgan_fingerprints
from qudock.kernels import KernelSVM, stack, tune_bandwidth
from qudock.quantum import _STATE_CACHE

log = logging.getLogger(__name__)

# measured on our own job: results/hardware/hardware_run.json
SECONDS_PER_CIRCUIT = 20 / 66
MEASURED_RUN = "ibm_marrakesh, 66 circuits, 1,024 shots, 20 s of quantum time"

KINDS = {"qsvc_pls": "fidelity", "hybrid_pls": "fidelity", "projected_pls": "projected",
         "tanimoto_svm": "classical", "rf_fingerprint": "classical", "svm_rbf_pls": "classical"}


def circuits_needed(kind: str, n_train: int, n_test: int) -> int:
    """Circuits a real device would run for this kernel, training and scoring together."""
    if kind == "fidelity":
        return n_train * (n_train + 1) // 2 + n_train * n_test
    if kind == "projected":
        return n_train + n_test
    return 0


def device_seconds(circuits: int) -> float:
    """Device time at the throughput of our own hardware run."""
    return circuits * SECONDS_PER_CIRCUIT


def _build(name: str, n_qubits: int, bandwidth: float, seed: int):
    from sklearn.ensemble import RandomForestClassifier

    from qudock.classical import make_classical

    if name == "rf_fingerprint":
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    if name == "svm_rbf_pls":
        return make_classical("svm_rbf", seed=seed, n_train=400)
    if name == "tanimoto_svm":
        return KernelSVM(kind="tanimoto", n_angles=n_qubits, C=1)
    if name == "projected_pls":
        return KernelSVM(kind="projected", n_angles=n_qubits, C=1, gamma=0.2,
                         bandwidth=bandwidth)
    kind = "hybrid" if name == "hybrid_pls" else "quantum"
    return KernelSVM(kind=kind, n_angles=n_qubits, C=1, weight=0.25, bandwidth=bandwidth)


def run_cost_study(sizes=(100, 200, 400, None), models=tuple(KINDS), n_qubits: int = 6,
                   seed: int = 0, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Time each model at several training sizes and count the circuits it would need."""
    df = load_bace() if df is None else df
    split = scaffold_split(df["smiles"])
    smiles, y = df["smiles"].values, df["label"].values
    rng = np.random.default_rng(seed)
    smi_te, y_te = smiles[split.test], y[split.test]
    fp_te = morgan_fingerprints(smi_te)

    rows = []
    for size in sizes:
        keep = (split.train if size is None else
                rng.choice(split.train, size=min(size, len(split.train)), replace=False))
        smi_tr, y_tr = smiles[keep], y[keep]
        fp_tr = morgan_fingerprints(smi_tr)
        pipe = FeaturePipeline(n_qubits, seed=seed, method="pls", use_descriptors=True)
        A_tr, A_te = pipe.fit_transform(smi_tr, y_tr), pipe.transform(smi_te)
        bandwidth = tune_bandwidth(A_tr, y_tr, n_qubits, seed=seed)
        for name in models:
            kind = KINDS[name]
            X_tr, X_te = (stack(A_tr, fp_tr), stack(A_te, fp_te))
            if name == "rf_fingerprint":
                X_tr, X_te = fp_tr, fp_te
            elif name == "svm_rbf_pls":
                X_tr, X_te = A_tr, A_te
            _STATE_CACHE.clear()        # time a cold run, the way a user would see it
            began = time.perf_counter()
            model = _build(name, n_qubits, bandwidth, seed).fit(X_tr, y_tr)
            fitted = time.perf_counter()
            score = (model.decision_function(X_te) if hasattr(model, "decision_function")
                     else model.predict_proba(X_te)[:, 1])
            done = time.perf_counter()
            circuits = circuits_needed(kind, len(keep), len(smi_te))
            rows.append({"model": name, "kind": kind, "n_train": len(keep),
                         "n_test": len(smi_te), "fit_seconds": fitted - began,
                         "score_seconds": done - fitted, "total_seconds": done - began,
                         "circuits": circuits, "device_seconds": device_seconds(circuits),
                         "device_hours": device_seconds(circuits) / 3600,
                         "roc_auc": float(roc_auc_score(y_te, score))})
            log.info("%s n=%d: %.1f s simulated, %d circuits, %.1f h on device", name, len(keep),
                     rows[-1]["total_seconds"], circuits, rows[-1]["device_hours"])
    return pd.DataFrame(rows)


def cost_table(results: pd.DataFrame) -> pd.DataFrame:
    """The full-size row of each model: accuracy next to what it cost to get it."""
    full = results[results["n_train"] == results["n_train"].max()]
    out = full[["model", "kind", "roc_auc", "total_seconds", "circuits", "device_hours"]].copy()
    return out.sort_values("roc_auc", ascending=False).reset_index(drop=True)
