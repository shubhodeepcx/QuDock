"""Kernels beyond the plain ZZ fidelity kernel.

* Tanimoto kernel on full fingerprints: the standard similarity in
  cheminformatics, and the strongest classical kernel on BACE.
* Projected quantum kernel (Huang et al., 2021): measure each qubit's
  Pauli expectations and put an RBF kernel on them. Local measurements resist
  the concentration that hits fidelity kernels as qubits grow, and hardware
  needs one circuit per molecule instead of one per pair.
* Hybrid kernel: a weighted sum of a quantum kernel and the Tanimoto kernel,
  so the quantum part only has to add what the fingerprint misses.
* Bandwidth tuning by kernel-target alignment, which needs no SVM fits.
"""

from __future__ import annotations

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC

from qudock.analysis import kernel_target_alignment
from qudock.quantum import ZZKernel


def tanimoto_kernel(A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
    """Tanimoto similarity between rows of two 0/1 fingerprint matrices."""
    A = np.asarray(A, dtype=np.float64)
    B = A if B is None else np.asarray(B, dtype=np.float64)
    inter = A @ B.T
    union = A.sum(1)[:, None] + B.sum(1)[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _pauli_ops(n_qubits: int) -> list[SparsePauliOp]:
    ops = []
    for q in range(n_qubits):
        for p in "XYZ":
            label = ["I"] * n_qubits
            label[n_qubits - 1 - q] = p  # qiskit orders qubits right to left
            ops.append(SparsePauliOp("".join(label)))
    return ops


class ProjectedKernel:
    """RBF kernel on single-qubit Pauli expectations of the ZZ feature map state.

    ``gamma`` follows Huang et al.: k(x, x') = exp(-gamma * sum_q ||rho_q(x) - rho_q(x')||^2),
    with each one-qubit density matrix described by its (X, Y, Z) expectations.
    """

    def __init__(self, n_qubits: int, reps: int = 2, bandwidth: float = 1.0, gamma: float = 1.0):
        self.zz = ZZKernel(n_qubits, reps=reps, bandwidth=bandwidth)
        self.gamma = gamma
        self._ops = [op.to_matrix(sparse=True) for op in _pauli_ops(n_qubits)]

    def features(self, X: np.ndarray) -> np.ndarray:
        """(n_samples, 3 * n_qubits) Pauli expectations: what hardware would measure."""
        sv = self.zz.statevectors(X)
        return np.stack([np.real(np.einsum("ij,ij->i", sv.conj(), (op @ sv.T).T))
                         for op in self._ops], axis=1)

    def from_features(self, F: np.ndarray, G: np.ndarray | None = None) -> np.ndarray:
        G = F if G is None else G
        sq = (F**2).sum(1)[:, None] + (G**2).sum(1)[None, :] - 2 * F @ G.T
        return np.exp(-self.gamma * np.clip(sq, 0, None))

    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        F = self.features(X)
        return self.from_features(F, None if Y is None else self.features(Y))


def tune_bandwidth(X: np.ndarray, y, n_qubits: int, reps: int = 2,
                   grid=(0.1, 0.2, 0.3, 0.5, 0.75, 1.0), max_samples: int = 300,
                   seed: int = 0) -> float:
    """Bandwidth with the best kernel-target alignment on (a sample of) the training set."""
    X, y = np.asarray(X), np.asarray(y)
    if len(X) > max_samples:
        idx = np.random.default_rng(seed).choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]
    scores = {b: kernel_target_alignment(ZZKernel(n_qubits, reps=reps, bandwidth=b)(X), y)
              for b in grid}
    return max(scores, key=scores.get)


class KernelSVM(ClassifierMixin, BaseEstimator):
    """SVM on a precomputed mix of kernels, usable inside GridSearchCV.

    Inputs are one matrix whose first ``n_angles`` columns are qubit angles and
    whose remaining columns are a 0/1 fingerprint (see ``stack``). ``kind``
    picks the kernel:

    * ``"quantum"``: ZZ fidelity kernel on the angles
    * ``"projected"``: projected quantum kernel on the angles
    * ``"tanimoto"``: Tanimoto kernel on the fingerprint
    * ``"hybrid"`` / ``"hybrid_projected"``: weight * quantum + (1 - weight) * Tanimoto
    """

    def __init__(self, kind: str = "quantum", n_angles: int = 6, C: float = 1.0,
                 weight: float = 0.5, bandwidth: float = 0.5, gamma: float = 1.0,
                 reps: int = 2, seed: int = 0):
        self.kind = kind
        self.n_angles = n_angles
        self.C = C
        self.weight = weight
        self.bandwidth = bandwidth
        self.gamma = gamma
        self.reps = reps
        self.seed = seed

    def _quantum(self):
        if self.kind in ("projected", "hybrid_projected"):
            return ProjectedKernel(self.n_angles, self.reps, self.bandwidth, self.gamma)
        return ZZKernel(self.n_angles, reps=self.reps, bandwidth=self.bandwidth)

    def _gram(self, X, Y=None):
        a, fa = X[:, :self.n_angles], X[:, self.n_angles:]
        b, fb = (None, None) if Y is None else (Y[:, :self.n_angles], Y[:, self.n_angles:])
        if self.kind == "tanimoto":
            return tanimoto_kernel(fa, fb)
        kq = self.kernel_(a, b)
        if self.kind in ("quantum", "projected"):
            return kq
        return self.weight * kq + (1 - self.weight) * tanimoto_kernel(fa, fb)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.kernel_ = self._quantum()
        self.X_train_ = X
        self.svc_ = SVC(kernel="precomputed", C=self.C, random_state=self.seed)
        self.svc_.fit(self._gram(X), y)
        self.classes_ = self.svc_.classes_
        return self

    def decision_function(self, X):
        return self.svc_.decision_function(self._gram(np.asarray(X, dtype=float), self.X_train_))

    def predict(self, X):
        return self.svc_.predict(self._gram(np.asarray(X, dtype=float), self.X_train_))


def stack(angles: np.ndarray, fingerprints: np.ndarray) -> np.ndarray:
    """Join qubit angles and fingerprints into the single input KernelSVM expects."""
    return np.hstack([angles, fingerprints])
