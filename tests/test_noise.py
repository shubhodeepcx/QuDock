import numpy as np

from qudock.noise import SampledZZKernel, kernel_error, nearest_psd
from qudock.quantum import ZZKernel

rng = np.random.default_rng(0)
X = rng.uniform(0, np.pi, size=(6, 2))


def test_sampled_kernel_approaches_exact_kernel():
    exact = ZZKernel(2, bandwidth=0.5)(X)
    sampled = SampledZZKernel(2, bandwidth=0.5, shots=20000)(X)
    assert np.allclose(sampled, sampled.T)
    assert np.allclose(np.diag(sampled), 1)
    assert np.abs(sampled - exact).max() < 0.03


def test_cross_kernel_shape():
    K = SampledZZKernel(2, shots=256)(X[:2], X)
    assert K.shape == (2, 6)
    assert ((K >= 0) & (K <= 1)).all()


def test_nearest_psd_removes_negative_eigenvalues():
    K = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.9], [0.1, 0.9, 1.0]])
    assert np.linalg.eigvalsh(K).min() < 0
    assert np.linalg.eigvalsh(nearest_psd(K)).min() > -1e-12


def test_kernel_error_is_zero_for_identical_kernels():
    K = ZZKernel(2)(X)
    err = kernel_error(K, K)
    assert np.isclose(err["rel_frobenius"], 0) and np.isclose(err["alignment_to_exact"], 1)
