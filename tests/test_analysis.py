import numpy as np
import pytest
from sklearn.metrics.pairwise import rbf_kernel

from qudock.analysis import (
    disagreements,
    effective_dimension,
    geometric_difference,
    kernel_target_alignment,
    off_diagonal_stats,
)

rng = np.random.default_rng(0)
X = rng.normal(size=(30, 3))


def test_geometric_difference_of_a_kernel_with_itself_is_about_one():
    K = rbf_kernel(X, gamma=0.5)
    assert np.isclose(geometric_difference(K, K, reg=1e-8), 1, atol=1e-2)


def test_identity_kernel_is_far_from_a_smooth_one():
    smooth = rbf_kernel(X, gamma=0.01)
    assert geometric_difference(smooth, np.eye(len(X))) > 5


def test_alignment_is_one_for_the_ideal_kernel():
    y = rng.integers(0, 2, 30)
    s = np.where(y > 0, 1.0, -1.0)
    assert np.isclose(kernel_target_alignment(np.outer(s, s), y), 1)


def test_effective_dimension_bounds():
    assert np.isclose(effective_dimension(np.eye(10)), 10)
    assert np.isclose(effective_dimension(np.ones((10, 10))), 1)


def test_off_diagonal_stats_and_disagreements():
    stats = off_diagonal_stats(np.eye(4))
    assert stats == {"mean": 0.0, "var": 0.0}
    d = disagreements([1, 0, 1, 0], [1, 0, 0, 0], [0, 0, 1, 0])
    assert list(d["only_a"]) == [0] and list(d["only_b"]) == [2]


def test_degenerate_kernels_do_not_divide_by_zero():
    from qudock.analysis import effective_dimension, kernel_target_alignment, normalise_trace

    zero = np.zeros((4, 4))
    assert kernel_target_alignment(zero, [0, 1, 0, 1]) == 0.0
    assert effective_dimension(zero) == 0.0
    with pytest.raises(ValueError, match="trace"):
        normalise_trace(zero)
