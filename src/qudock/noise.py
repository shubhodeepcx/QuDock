"""How much of the ideal ZZ kernel survives shot noise and hardware noise?

Each kernel entry k(x, y) is estimated the way real hardware would do it: run
the compute-uncompute circuit U(y)^dagger U(x) and count how often every qubit
comes back as 0. We compare against the exact statevector kernel.
"""

from __future__ import annotations

import numpy as np
from qiskit import transpile
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.circuit.library import zz_feature_map
from qiskit_aer import AerSimulator


def fidelity_circuit(n_qubits: int, reps: int = 2, entanglement: str = "linear"):
    """Parameterised U(y)^dagger U(x) with measurements, plus its x and y parameters."""
    x = ParameterVector("x", n_qubits)
    y = ParameterVector("y", n_qubits)
    fm = zz_feature_map(n_qubits, reps=reps, entanglement=entanglement)
    qc = QuantumCircuit(n_qubits)
    qc.compose(fm.assign_parameters(x), inplace=True)
    qc.compose(fm.assign_parameters(y).inverse(), inplace=True)
    qc.measure_all()
    return qc, x, y


def make_simulator(backend=None, seed: int = 0) -> AerSimulator:
    """Noise-free sampler, or one carrying a fake IBM backend's noise model."""
    if backend is None:
        return AerSimulator(seed_simulator=seed)
    return AerSimulator.from_backend(backend, seed_simulator=seed)


class SampledZZKernel:
    """ZZ fidelity kernel estimated from measurement counts."""

    def __init__(self, n_qubits: int, reps: int = 2, entanglement: str = "linear",
                 bandwidth: float = 1.0, shots: int = 1024, simulator=None, seed: int = 0):
        self.n_qubits = n_qubits
        self.bandwidth = bandwidth
        self.shots = shots
        self.seed = seed
        self.simulator = simulator or make_simulator(seed=seed)
        circuit, self._x, self._y = fidelity_circuit(n_qubits, reps, entanglement)
        # transpile once (level 3 picks a good qubit layout); each entry only binds new angles
        self.circuit = transpile(circuit, self.simulator, optimization_level=3,
                                 seed_transpiler=seed)

    def _estimate(self, pairs: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        circuits = [
            self.circuit.assign_parameters({**dict(zip(self._x, self.bandwidth * a)),
                                            **dict(zip(self._y, self.bandwidth * b))})
            for a, b in pairs
        ]
        result = self.simulator.run(circuits, shots=self.shots).result()
        zeros = "0" * self.n_qubits
        return np.array([result.get_counts(i).get(zeros, 0) / self.shots
                         for i in range(len(circuits))])

    def __call__(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if Y is None:
            # symmetric: estimate the upper triangle only, diagonal is 1 by definition
            iu = np.triu_indices(len(X), k=1)
            K = np.eye(len(X))
            K[iu] = self._estimate([(X[i], X[j]) for i, j in zip(*iu)])
            return K + np.triu(K, 1).T
        Y = np.asarray(Y, dtype=float)
        values = self._estimate([(x, y) for x in X for y in Y])
        return values.reshape(len(X), len(Y))


def nearest_psd(K: np.ndarray) -> np.ndarray:
    """Clip negative eigenvalues so a noisy Gram matrix is a valid kernel again."""
    lam, V = np.linalg.eigh((K + K.T) / 2)
    return (V * np.clip(lam, 0, None)) @ V.T


def kernel_error(K_noisy: np.ndarray, K_exact: np.ndarray) -> dict[str, float]:
    """Relative Frobenius error and alignment between a noisy and an exact kernel."""
    diff = np.linalg.norm(K_noisy - K_exact) / np.linalg.norm(K_exact)
    align = np.sum(K_noisy * K_exact) / (np.linalg.norm(K_noisy) * np.linalg.norm(K_exact))
    return {"rel_frobenius": float(diff), "alignment_to_exact": float(align)}
