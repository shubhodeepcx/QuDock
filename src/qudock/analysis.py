"""Where could the quantum kernel help? Kernel geometry diagnostics.

* geometric difference (Huang et al., "Power of data in quantum machine
  learning", 2021): if g(K_C || K_Q) is small, a classical model with kernel
  K_C can match the quantum kernel on any labels, so no advantage is possible.
* kernel-target alignment: how well a kernel's geometry matches the labels.
* concentration: off-diagonal kernel values bunch up as qubits grow, which
  makes the kernel close to the identity and hard to learn from.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm


def normalise_trace(K: np.ndarray) -> np.ndarray:
    """Scale a Gram matrix so its trace equals its size, as in Huang et al."""
    trace = np.trace(K)
    if trace <= 0:
        raise ValueError("cannot normalise a Gram matrix with a non-positive trace")
    return K * (len(K) / trace)


def geometric_difference(K_classical: np.ndarray, K_quantum: np.ndarray,
                         reg: float = 1e-3) -> float:
    """g(K_C || K_Q) = sqrt(|| sqrt(K_Q) (K_C + reg I)^-1 sqrt(K_Q) ||_inf).

    Large values (around sqrt(N)) mean the quantum kernel holds structure the
    classical one can't reproduce; values near 1 mean it doesn't.
    """
    Kc, Kq = normalise_trace(K_classical), normalise_trace(K_quantum)
    sq = np.real(sqrtm(Kq))
    inner = sq @ np.linalg.inv(Kc + reg * np.eye(len(Kc))) @ sq
    return float(np.sqrt(np.max(np.linalg.eigvalsh((inner + inner.T) / 2))))


def kernel_target_alignment(K: np.ndarray, y) -> float:
    """<K, yy^T>_F / (||K||_F ||yy^T||_F), with labels mapped to +/-1."""
    y = np.where(np.asarray(y) > 0, 1.0, -1.0)
    Y = np.outer(y, y)
    norm = np.linalg.norm(K) * np.linalg.norm(Y)
    return float(np.sum(K * Y) / norm) if norm > 0 else 0.0


def effective_dimension(K: np.ndarray) -> float:
    """Participation ratio of the eigenspectrum: (sum l)^2 / sum l^2."""
    lam = np.clip(np.linalg.eigvalsh(K), 0, None)
    total = np.sum(lam**2)
    return float(lam.sum() ** 2 / total) if total > 0 else 0.0


def off_diagonal_stats(K: np.ndarray) -> dict[str, float]:
    """Mean and variance of off-diagonal entries; variance -> 0 means concentration."""
    off = K[~np.eye(len(K), dtype=bool)]
    return {"mean": float(off.mean()), "var": float(off.var())}


def disagreements(y_true, pred_a, pred_b) -> dict[str, np.ndarray]:
    """Indices where exactly one of two models is right."""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    return {"only_a": np.flatnonzero(a_ok & ~b_ok), "only_b": np.flatnonzero(b_ok & ~a_ok)}
