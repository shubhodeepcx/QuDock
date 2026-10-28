import numpy as np
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector
from sklearn.metrics import roc_auc_score

from qudock.quantum import QSVC, ZZKernel

rng = np.random.default_rng(0)


def test_kernel_is_a_valid_gram_matrix():
    X = rng.uniform(0, np.pi, size=(12, 3))
    K = ZZKernel(3)(X)
    assert np.allclose(np.diag(K), 1)
    assert np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-9
    assert K.min() >= 0 and K.max() <= 1 + 1e-9


def test_kernel_matches_direct_fidelity():
    x, y = rng.uniform(0, np.pi, size=(2, 2))
    fm = zz_feature_map(2, reps=2, entanglement="linear")
    a = Statevector(fm.assign_parameters(x))
    b = Statevector(fm.assign_parameters(y))
    expected = abs(a.inner(b)) ** 2
    assert np.isclose(ZZKernel(2)(x[None], y[None])[0, 0], expected)


def test_statevectors_are_cached_across_instances():
    X = rng.uniform(0, np.pi, size=(5, 2))
    first = ZZKernel(2, bandwidth=0.37)
    first(X)
    second = ZZKernel(2, bandwidth=0.37)
    assert second._cache is first._cache
    assert len(second._cache) == 5


def test_bandwidth_zero_gives_constant_kernel():
    X = rng.uniform(0, np.pi, size=(4, 2))
    assert np.allclose(ZZKernel(2, bandwidth=0.0)(X), 1)


def test_qsvc_learns_a_separable_problem():
    X = rng.uniform(0, np.pi, size=(60, 2))
    y = (X[:, 0] > np.pi / 2).astype(int)
    model = QSVC(C=10, reps=1, bandwidth=0.5).fit(X[:40], y[:40])
    assert roc_auc_score(y[40:], model.decision_function(X[40:])) > 0.9


def test_state_cache_can_be_cleared():
    from qudock.quantum import clear_state_cache, state_cache_size

    ZZKernel(2, bandwidth=0.123)(np.array([[0.1, 0.2], [0.3, 0.4]]))
    assert state_cache_size() >= 2
    clear_state_cache()
    assert state_cache_size() == 0
