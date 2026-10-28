import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from qudock.classical import GRIDS, make_classical, scores


@pytest.mark.parametrize("name", list(GRIDS))
def test_baselines_learn_a_separable_problem(name):
    rng = np.random.default_rng(0)
    X = rng.uniform(0, np.pi, size=(80, 3))
    y = (X[:, 0] > np.pi / 2).astype(int)
    model = make_classical(name, n_train=len(y)).fit(X[:60], y[:60])
    assert roc_auc_score(y[60:], scores(model, X[60:])) > 0.9


def test_unknown_model_name_lists_the_choices():
    with pytest.raises(ValueError, match="random_forest"):
        make_classical("xgboost")
