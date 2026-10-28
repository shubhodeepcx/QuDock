import numpy as np
import pytest
from sklearn.svm import SVC

from qudock.conformal import ConformalClassifier, conformal_report, describe_set

rng = np.random.default_rng(0)


def _toy(n=400, gap=1.0):
    y = rng.integers(0, 2, n)
    X = (y * gap + rng.normal(0, 1, n))[:, None]
    return X, y


def test_p_values_are_between_zero_and_one():
    X, y = _toy()
    p = ConformalClassifier(SVC()).fit(X, y).p_values(X)
    assert p.shape == (len(y), 2)
    assert np.all((p > 0) & (p <= 1))


def test_coverage_matches_the_requested_confidence():
    X, y = _toy(n=600)
    cc = ConformalClassifier(SVC(), seed=1).fit(X[:400], y[:400])
    for confidence in (0.7, 0.9):
        report = conformal_report(y[400:], cc.predict_set(X[400:], confidence))
        for key in ("coverage_active", "coverage_inactive"):
            assert report[key] > confidence - 0.1, (confidence, key)


def test_higher_confidence_never_shrinks_the_prediction_set():
    X, y = _toy()
    cc = ConformalClassifier(SVC()).fit(X, y)
    assert np.all(cc.predict_set(X, 0.95).sum(axis=1) >= cc.predict_set(X, 0.7).sum(axis=1))


def test_a_single_label_is_right_on_an_easy_problem():
    X, y = _toy(gap=6.0)
    report = conformal_report(y, ConformalClassifier(SVC()).fit(X, y).predict_set(X, 0.8))
    assert report["single_label_accuracy"] == 1.0


def _share_of_dont_know(gap):
    X, y = _toy(gap=gap)
    sets = ConformalClassifier(SVC()).fit(X, y).predict_set(X, 0.9)
    return (sets.sum(axis=1) == 2).mean()


def test_a_hard_problem_says_it_does_not_know_more_often():
    assert _share_of_dont_know(0.2) > _share_of_dont_know(6.0)


def test_describe_set_names_every_case():
    assert describe_set([False, True]) == "active"
    assert describe_set([True, False]) == "inactive"
    assert describe_set([True, True]) == "don't know"
    assert describe_set([False, False]) == "unlike anything seen"


def test_confidence_study_runs_end_to_end_on_toy_data():
    import pandas as pd

    from qudock.conformal import run_confidence_study

    # several scaffold families, so the test and calibration splits both have chemistry of
    # their own and both labels
    scaffolds = ["c1ccccc1", "C1CCNCC1", "c1ccncc1", "C1CCOC1", "c1ccc2ccccc2c1", "C1CCCCC1"]
    subs = ["O", "N", "C", "Cl", "F", "Br", "CO", "CN"]
    toy = pd.DataFrame({"smiles": [s + sub for s in scaffolds for sub in subs],
                        "label": [1, 0] * (len(scaffolds) * len(subs) // 2)})
    sets, domain = run_confidence_study(n_qubits=2, df=toy)
    assert set(sets["model"]) == {"qsvc_pls", "hybrid_pls", "tanimoto_svm"}
    assert sets["coverage"].between(0, 1).all()
    assert len(domain) == 9


def test_works_with_models_that_only_give_probabilities():
    from sklearn.ensemble import RandomForestClassifier

    X, y = _toy()
    cc = ConformalClassifier(RandomForestClassifier(n_estimators=20, random_state=0)).fit(X, y)
    assert cc.predict_set(X[:5]).shape == (5, 2)


def test_bad_fractions_are_refused():
    X, y = _toy()
    with pytest.raises(ValueError, match="calib_frac"):
        ConformalClassifier(SVC(), calib_frac=1.2).fit(X, y)
    with pytest.raises(ValueError, match="confidence"):
        ConformalClassifier(SVC()).fit(X, y).predict_set(X, confidence=90)
