import numpy as np
import pytest

from qudock.benchmark import bootstrap_ci
from qudock.screening import paired_bootstrap, screening_metrics


def test_bootstrap_needs_both_classes():
    y, s = np.ones(10, dtype=int), np.linspace(0, 1, 10)
    with pytest.raises(ValueError, match="both labels"):
        paired_bootstrap(y, s, s, n_boot=20)
    with pytest.raises(ValueError, match="both labels"):
        bootstrap_ci(y, s, n_boot=20)


def test_perfect_ranking_gets_perfect_screening_metrics():
    y = np.array([1] * 10 + [0] * 90)
    m = screening_metrics(y, -np.arange(100))
    assert m["roc_auc"] == pytest.approx(1.0) and m["pr_auc"] == pytest.approx(1.0)
    assert m["bedroc"] == pytest.approx(1.0, abs=1e-3)
    assert m["ef_5"] == pytest.approx(10.0)
