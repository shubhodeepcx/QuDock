"""Quantum kernel SVM on a ZZFeatureMap fidelity kernel.

The kernel is k(x, y) = |<phi(x)|phi(y)>|^2 where |phi(x)> is the
ZZFeatureMap state. In ideal simulation we compute each statevector once and
take inner products, which is much faster than running a fidelity circuit per
pair. Statevectors are cached so hyperparameter sweeps never recompute them.
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC

# statevectors shared across kernel instances, keyed by feature-map settings,
# so refitting inside GridSearchCV or a learning curve reuses earlier work
_STATE_CACHE: dict[tuple, dict[bytes, np.ndarray]] = {}


def clear_state_cache() -> None:
    """Forget every cached statevector, e.g. before switching to a new qubit count."""
    _STATE_CACHE.clear()


def state_cache_size() -> int:
    """Number of statevectors currently held in memory."""
    return sum(len(states) for states in _STATE_CACHE.values())


class ZZKernel:
    """Exact fidelity kernel for a ZZFeatureMap.

    ``bandwidth`` scales the input angles (Shaydulin & Wild, 2022); values
    below 1 stop the kernel collapsing towards the identity as qubits grow.
    """

    def __init__(self, n_qubits: int, reps: int = 2, entanglement: str = "linear",
                 bandwidth: float = 1.0):
        self.n_qubits = n_qubits
        self.reps = reps
        self.entanglement = entanglement
        self.bandwidth = bandwidth
        self.circuit = zz_feature_map(n_qubits, reps=reps, entanglement=entanglement)
        key = (n_qubits, reps, entanglement, float(bandwidth))
        self._cache = _STATE_CACHE.setdefault(key, {})

    def _simulate(self, X: np.ndarray) -> np.ndarray:
        """Statevectors of the bound feature map, in Qiskit's qubit order."""
        if self.entanglement != "linear":
            return np.array([Statevector(self.circuit.assign_parameters(self.bandwidth * x)).data
                             for x in X])
        # the trainable kernel's numpy simulator with its weights left at one is
        # this exact map, batched over molecules instead of one circuit each
        from qudock.trainable import TrainableKernel

        states = TrainableKernel(self.n_qubits, self.reps,
                                 bandwidth=self.bandwidth).statevectors(X)
        # it numbers qubits from the left, Qiskit from the right; global phase may differ
        n = self.n_qubits
        axes = (0, *range(n, 0, -1))
        return states.reshape((len(X),) + (2,) * n).transpose(axes).reshape(len(X), -1)

    def statevectors(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        keys = [x.tobytes() for x in X]
        missing = {k: i for i, k in enumerate(keys) if k not in self._cache}
        if missing:
            for k, state in zip(missing, self._simulate(X[list(missing.values())])):
                self._cache[k] = state
        out = np.empty((len(X), 2**self.n_qubits), dtype=complex)
        for i, k in enumerate(keys):
            out[i] = self._cache[k]
        return out

    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        sx = self.statevectors(X)
        sy = sx if Y is None else self.statevectors(Y)
        return np.abs(sx.conj() @ sy.T) ** 2


class QSVC(ClassifierMixin, BaseEstimator):
    """SVM on a precomputed ZZ fidelity kernel, usable inside GridSearchCV."""

    def __init__(self, C: float = 1.0, reps: int = 2, entanglement: str = "linear",
                 bandwidth: float = 1.0, seed: int = 0):
        self.C = C
        self.reps = reps
        self.entanglement = entanglement
        self.bandwidth = bandwidth
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.kernel_ = ZZKernel(X.shape[1], self.reps, self.entanglement, self.bandwidth)
        self.X_train_ = X
        self.svc_ = SVC(kernel="precomputed", C=self.C, random_state=self.seed)
        self.svc_.fit(self.kernel_(X), y)
        self.classes_ = self.svc_.classes_
        return self

    def decision_function(self, X):
        return self.svc_.decision_function(self.kernel_(np.asarray(X, dtype=float), self.X_train_))

    def predict(self, X):
        return self.svc_.predict(self.kernel_(np.asarray(X, dtype=float), self.X_train_))
