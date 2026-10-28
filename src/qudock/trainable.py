"""A feature map whose own parameters are trained (Hubregtsen et al., 2022).

The ZZ map is fixed: the only thing we could tune was one global bandwidth.
Here every encoding angle gets its own weight and every qubit a trainable
rotation, and the parameters are chosen by maximising kernel-target alignment,
which needs no SVM fits and no labels beyond the training set.

Per layer the circuit does Hadamards, RZ(2 * w_q * x_q) on each qubit, a ZZ rotation
by 2 * v_q * (pi - x_q)(pi - x_q+1) between neighbours, then RY(theta_q).
With all w = v = 1 and no RY this is exactly the ZZ feature map, so the fixed
map is the starting point of the search, and any gain is the training's.

Statevectors are simulated directly with numpy: one qubit's gate is a product
on a reshaped axis, so a few hundred molecules at 6 qubits cost milliseconds
and the whole optimisation runs on a laptop CPU.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize

from qudock.analysis import kernel_target_alignment

log = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def _z_signs(n_qubits: int) -> tuple[np.ndarray, np.ndarray]:
    """+/-1 eigenvalues of each Z_q and of each neighbouring Z_q Z_q+1, per basis state."""
    bits = np.arange(2**n_qubits)
    z = np.stack([1 - 2 * ((bits >> (n_qubits - 1 - q)) & 1) for q in range(n_qubits)])
    return z.astype(float), (z[:-1] * z[1:]).astype(float)


def _apply_diagonal(state: np.ndarray, n_qubits: int, rz: np.ndarray, zz: np.ndarray):
    """Every RZ and neighbouring ZZ rotation of a layer at once.

    All of them are diagonal and commute, so their phases simply add up and one
    complex exponential replaces ``2n - 1`` separate ones. ``rz`` is
    (n_samples, n_qubits) and ``zz`` is (n_samples, n_qubits - 1).
    """
    z, zz_sign = _z_signs(n_qubits)
    phase = rz @ z + (zz @ zz_sign if n_qubits > 1 else 0.0)
    return state * np.exp(-0.5j * phase)


def _apply_h(state: np.ndarray, qubit: int, n_qubits: int):
    view = state.reshape(len(state), 2**qubit, 2, 2 ** (n_qubits - 1 - qubit))
    a, b = view[:, :, 0, :].copy(), view[:, :, 1, :].copy()
    view[:, :, 0, :] = (a + b) / np.sqrt(2)
    view[:, :, 1, :] = (a - b) / np.sqrt(2)
    return view.reshape(len(state), -1)


def _apply_ry(state: np.ndarray, qubit: int, n_qubits: int, theta: float):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    view = state.reshape(len(state), 2**qubit, 2, 2 ** (n_qubits - 1 - qubit))
    a, b = view[:, :, 0, :].copy(), view[:, :, 1, :].copy()
    view[:, :, 0, :] = c * a - s * b
    view[:, :, 1, :] = s * a + c * b
    return view.reshape(len(state), -1)


@dataclass
class KernelParams:
    """Encoding weights per qubit, entangling weights per pair, and RY angles."""
    encode: np.ndarray          # (reps, n_qubits)
    entangle: np.ndarray        # (reps, n_qubits - 1)
    rotate: np.ndarray          # (reps, n_qubits)

    @classmethod
    def fixed(cls, n_qubits: int, reps: int = 2) -> KernelParams:
        """The plain ZZ feature map: unit weights, no extra rotation."""
        return cls(np.ones((reps, n_qubits)), np.ones((reps, n_qubits - 1)),
                   np.zeros((reps, n_qubits)))

    def flat(self) -> np.ndarray:
        return np.concatenate([self.encode.ravel(), self.entangle.ravel(), self.rotate.ravel()])

    @classmethod
    def from_flat(cls, flat: np.ndarray, n_qubits: int, reps: int) -> KernelParams:
        a, b = reps * n_qubits, reps * (n_qubits - 1)
        return cls(flat[:a].reshape(reps, n_qubits),
                   flat[a:a + b].reshape(reps, n_qubits - 1),
                   flat[a + b:].reshape(reps, n_qubits))


class TrainableKernel:
    """Fidelity kernel of the parametrised feature map above."""

    def __init__(self, n_qubits: int, reps: int = 2, params: KernelParams | None = None,
                 bandwidth: float = 1.0):
        self.n_qubits, self.reps, self.bandwidth = n_qubits, reps, bandwidth
        self.params = params or KernelParams.fixed(n_qubits, reps)

    def statevectors(self, X: np.ndarray) -> np.ndarray:
        X = self.bandwidth * np.asarray(X, dtype=float)
        n, p = self.n_qubits, self.params
        state = np.full((len(X), 2**n), 2 ** (-n / 2), dtype=complex)
        for layer in range(self.reps):
            if layer:                       # the ZZ map re-applies Hadamards each repetition
                for q in range(n):
                    state = _apply_h(state, q, n)
            rz = 2 * p.encode[layer] * X
            zz = 2 * p.entangle[layer] * (np.pi - X[:, :-1]) * (np.pi - X[:, 1:])
            state = _apply_diagonal(state, n, rz, zz)
            for q in range(n):
                if p.rotate[layer, q]:
                    state = _apply_ry(state, q, n, p.rotate[layer, q])
        return state

    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        sx = self.statevectors(X)
        sy = sx if Y is None else self.statevectors(Y)
        return np.abs(sx.conj() @ sy.T) ** 2


def train_kernel(X, y, n_qubits: int, reps: int = 2, bandwidth: float = 1.0,
                 max_samples: int = 200, max_iter: int = 60, seed: int = 0):
    """Maximise kernel-target alignment over the feature map's own parameters.

    Returns the trained parameters and the alignment before and after, measured
    on the same molecules, so the gain from training is visible on its own.
    """
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    if len(X) > max_samples:
        idx = np.random.default_rng(seed).choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]

    def alignment(flat):
        params = KernelParams.from_flat(flat, n_qubits, reps)
        K = TrainableKernel(n_qubits, reps, params, bandwidth)(X)
        return kernel_target_alignment(K, y)

    start = KernelParams.fixed(n_qubits, reps).flat()
    before = alignment(start)
    result = minimize(lambda f: -alignment(f), start, method="Powell",
                      options={"maxiter": max_iter, "xtol": 1e-2, "ftol": 1e-3})
    after = -result.fun
    log.info("alignment %.4f -> %.4f in %d evaluations", before, after, result.nfev)
    if after <= before:            # keep the fixed map if training found nothing
        return KernelParams.fixed(n_qubits, reps), before, before
    return KernelParams.from_flat(result.x, n_qubits, reps), before, after


def run_trainable_study(sizes=(100, 200, 400, None), seeds=(0, 1, 2, 3, 4), n_qubits: int = 6,
                        reps: int = 2, max_samples: int = 150, max_iter: int = 30, df=None):
    """Fixed ZZ map against the same map with its parameters trained by alignment."""
    import pandas as pd
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.svm import SVC

    from qudock.benchmark import subsample
    from qudock.data import load_bace, scaffold_split
    from qudock.features import FeaturePipeline
    from qudock.kernels import tune_bandwidth
    from qudock.screening import screening_metrics

    df = load_bace() if df is None else df
    split = scaffold_split(df["smiles"])
    smiles, y = df["smiles"].values, df["label"].values
    smi_tr, y_tr = smiles[split.train], y[split.train]
    smi_te, y_te = smiles[split.test], y[split.test]

    rows = []
    for size in sizes:
        for seed in seeds:
            keep = subsample(y_tr, size, seed)
            pipe = FeaturePipeline(n_qubits, seed=seed, method="pls", use_descriptors=True)
            A_tr = pipe.fit_transform(smi_tr[keep], y_tr[keep])
            A_te = pipe.transform(smi_te)
            bandwidth = tune_bandwidth(A_tr, y_tr[keep], n_qubits, reps, seed=seed)
            start = time.perf_counter()
            params, before, after = train_kernel(A_tr, y_tr[keep], n_qubits, reps, bandwidth,
                                                 max_samples, max_iter, seed)
            seconds = time.perf_counter() - start
            for name, p, align in [("fixed", KernelParams.fixed(n_qubits, reps), before),
                                   ("trained", params, after)]:
                kernel = TrainableKernel(n_qubits, reps, p, bandwidth)
                cv = StratifiedKFold(5 if len(keep) >= 100 else 3, shuffle=True,
                                     random_state=seed)
                search = GridSearchCV(SVC(kernel="precomputed", random_state=seed),
                                      {"C": [0.1, 1, 10]}, cv=cv, scoring="roc_auc")
                search.fit(kernel(A_tr), y_tr[keep])
                score = search.best_estimator_.decision_function(kernel(A_te, A_tr))
                rows.append({"feature_map": name, "n_train": len(keep), "seed": seed,
                             "bandwidth": bandwidth, "alignment": align,
                             "C": search.best_params_["C"],
                             "training_seconds": seconds if name == "trained" else 0.0,
                             **screening_metrics(y_te, score)})
            log.info("n=%d seed=%d fixed %.3f trained %.3f", len(keep), seed,
                     rows[-2]["roc_auc"], rows[-1]["roc_auc"])
    return pd.DataFrame(rows)


def summarise_trainable(results):
    """Mean and 95% interval over seeds, per feature map and training size."""
    metrics = ["roc_auc", "pr_auc", "ef_1", "ef_5", "bedroc", "alignment"]
    g = results.groupby(["n_train", "feature_map"])[metrics]
    out = g.mean()
    out["roc_auc_ci"] = 1.96 * g.sem().fillna(0)["roc_auc"]
    return out.reset_index()
