import numpy as np
import pytest

from qudock.kernels import ProjectedKernel
from qudock.shadows import ShadowFeatures, estimate, exact_expectations, snapshots

rng = np.random.default_rng(0)
X = rng.uniform(0, np.pi, size=(4, 3))


def test_exact_one_body_terms_match_the_projected_kernel():
    exact = ShadowFeatures(3, bandwidth=0.7, shots=None)(X)
    assert np.allclose(exact, ProjectedKernel(3, bandwidth=0.7).features(X))


def test_two_body_terms_are_valid_correlators():
    F = ShadowFeatures(3, shots=None, two_body=True)(X)
    assert F.shape == (4, 9 + 9 * 3)
    assert np.all(np.abs(F) <= 1 + 1e-9)


def test_product_state_correlators_factorise():
    # |+>|0>: <X0> = 1, <Z1> = 1, <X0 Z1> = 1, everything else in that pair 0
    state = np.kron([1, 0], [1, 1]) / np.sqrt(2)      # qubit 0 is the right-hand factor
    e = exact_expectations(state, 2, two_body=True)
    assert e[0] == pytest.approx(1) and e[5] == pytest.approx(1)
    assert e[6 + 0 * 3 + 2] == pytest.approx(1)       # X on qubit 0, Z on qubit 1


@pytest.mark.parametrize("two_body", [False, True])
def test_shadow_estimates_converge_to_the_exact_values(two_body):
    state = ShadowFeatures(3, bandwidth=0.6).zz.statevectors(X[:1])[0]
    exact = exact_expectations(state, 3, two_body)
    est = estimate(*snapshots(state, 3, 60_000, np.random.default_rng(1)), two_body)
    assert np.max(np.abs(est - exact)) < 0.08


def test_fewer_shots_means_more_error():
    exact = ShadowFeatures(3, shots=None)(X)
    err = [np.abs(ShadowFeatures(3, shots=s, seed=2)(X) - exact).mean() for s in (100, 10_000)]
    assert err[0] > err[1]
