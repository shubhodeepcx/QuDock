"""Classical shadows: Pauli expectations of the feature-map state from random measurements.

Each shot measures every qubit in a uniformly random basis (X, Y or Z). For a
single-qubit Pauli P on qubit q, the snapshot estimate is 3 * b_q when that
qubit was measured in P's basis and 0 otherwise (b_q = +/-1 is the outcome);
for a two-qubit P_q P_r it is 9 * b_q * b_r when both bases match. Averaging
snapshots gives unbiased estimates of every such expectation at once (Huang,
Kueng & Preskill, 2020).

Single-qubit expectations alone need only three measurement settings, so the
point of shadows here is the second order: the same shots also give all
9 * n(n-1)/2 two-qubit correlations, which describe entanglement the
single-qubit projected kernel cannot see. ``ShadowFeatures`` simulates the
whole protocol from exact statevectors with a finite shot budget, so the
effect of sampling noise on the kernel can be measured before any hardware run.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from qudock.quantum import ZZKernel

PAULIS = "XYZ"
_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
_SDG = np.diag([1, -1j])
# unitary applied before a Z measurement to measure in each basis
BASIS_CHANGE = {0: _H, 1: _H @ _SDG, 2: np.eye(2)}


def _rotate(state: np.ndarray, bases: np.ndarray, n_qubits: int) -> np.ndarray:
    """Apply each qubit's basis change to a Qiskit-ordered statevector."""
    psi = state.reshape((2,) * n_qubits)
    for q, b in enumerate(bases):
        if b == 2:
            continue
        axis = n_qubits - 1 - q            # Qiskit: qubit 0 is the last tensor axis
        psi = np.moveaxis(np.tensordot(BASIS_CHANGE[int(b)], psi, axes=([1], [axis])), 0, axis)
    return psi.reshape(-1)


def snapshots(state: np.ndarray, n_qubits: int, shots: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Random bases (shots, n) in {0: X, 1: Y, 2: Z} and +/-1 outcomes (shots, n)."""
    bases = rng.integers(0, 3, size=(shots, n_qubits))
    outcomes = np.empty((shots, n_qubits), dtype=np.int8)
    settings, inverse = np.unique(bases, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    bits = np.arange(2**n_qubits)
    for k, setting in enumerate(settings):
        rows = np.flatnonzero(inverse == k)
        probs = np.abs(_rotate(state, setting, n_qubits)) ** 2
        drawn = rng.choice(bits, size=len(rows), p=probs / probs.sum())
        for q in range(n_qubits):
            outcomes[rows, q] = 1 - 2 * ((drawn >> q) & 1)
    return bases, outcomes


def estimate(bases: np.ndarray, outcomes: np.ndarray, two_body: bool = False) -> np.ndarray:
    """Shadow estimates: (X, Y, Z) per qubit, then 9 correlators per qubit pair if asked."""
    n = bases.shape[1]
    one = np.empty(3 * n)
    for q in range(n):
        for p in range(3):
            one[3 * q + p] = np.mean(3.0 * (bases[:, q] == p) * outcomes[:, q])
    if not two_body:
        return one
    two = []
    for q, r in combinations(range(n), 2):
        for p in range(3):
            for s in range(3):
                hit = (bases[:, q] == p) & (bases[:, r] == s)
                two.append(np.mean(9.0 * hit * outcomes[:, q] * outcomes[:, r]))
    return np.concatenate([one, two])


def exact_expectations(state: np.ndarray, n_qubits: int, two_body: bool = False) -> np.ndarray:
    """The quantities ``estimate`` targets, computed exactly, in the same order."""
    psi = state.reshape((2,) * n_qubits)
    paulis = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]

    def expect(ops: dict[int, np.ndarray]) -> float:
        phi = psi
        for q, op in ops.items():
            axis = n_qubits - 1 - q
            phi = np.moveaxis(np.tensordot(op, phi, axes=([1], [axis])), 0, axis)
        return float(np.real(np.vdot(psi, phi)))

    out = [expect({q: paulis[p]}) for q in range(n_qubits) for p in range(3)]
    if two_body:
        out += [expect({q: paulis[p], r: paulis[s]})
                for q, r in combinations(range(n_qubits), 2) for p in range(3) for s in range(3)]
    return np.array(out)


class ShadowFeatures:
    """Projected-kernel features of the ZZ feature-map state, estimated from shadows.

    ``shots=None`` returns the exact expectations, the infinite-shot limit.
    """

    def __init__(self, n_qubits: int, reps: int = 2, bandwidth: float = 1.0,
                 shots: int | None = 1000, two_body: bool = False, seed: int = 0):
        self.zz = ZZKernel(n_qubits, reps=reps, bandwidth=bandwidth)
        self.n_qubits = n_qubits
        self.shots = shots
        self.two_body = two_body
        self.seed = seed

    def __call__(self, X: np.ndarray) -> np.ndarray:
        states = self.zz.statevectors(X)
        if self.shots is None:
            return np.stack([exact_expectations(s, self.n_qubits, self.two_body)
                             for s in states])
        rng = np.random.default_rng(self.seed)
        return np.stack([estimate(*snapshots(s, self.n_qubits, self.shots, rng), self.two_body)
                         for s in states])
