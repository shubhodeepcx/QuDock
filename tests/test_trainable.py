import numpy as np
import pandas as pd

from qudock.quantum import ZZKernel
from qudock.trainable import (
    KernelParams,
    TrainableKernel,
    run_trainable_study,
    summarise_trainable,
    train_kernel,
)

rng = np.random.default_rng(0)


def test_untrained_map_is_exactly_the_zz_feature_map():
    X = rng.uniform(0, np.pi, size=(6, 4))
    assert np.allclose(TrainableKernel(4, 2)(X), ZZKernel(4, reps=2, bandwidth=1.0)(X))


def test_kernel_stays_a_valid_gram_matrix_with_trained_parameters():
    X = rng.uniform(0, np.pi, size=(8, 3))
    params = KernelParams(rng.normal(1, 0.3, (2, 3)), rng.normal(1, 0.3, (2, 2)),
                          rng.uniform(0, np.pi, (2, 3)))
    K = TrainableKernel(3, 2, params)(X)
    assert np.allclose(np.diag(K), 1)
    assert np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-9


def test_parameters_survive_a_round_trip_through_flat():
    params = KernelParams.fixed(5, 3)
    back = KernelParams.from_flat(params.flat(), 5, 3)
    assert np.allclose(back.encode, params.encode)
    assert np.allclose(back.entangle, params.entangle)
    assert np.allclose(back.rotate, params.rotate)


def test_training_improves_alignment_on_a_separable_problem():
    X = rng.uniform(0, np.pi, size=(40, 3))
    y = (X[:, 0] > np.pi / 2).astype(int)
    _, before, after = train_kernel(X, y, 3, max_iter=5, max_samples=40)
    assert after >= before


def _toy():
    scaffolds = ["c1ccccc1", "C1CCNCC1", "c1ccncc1", "C1CCOC1"]
    subs = ["O", "N", "C", "Cl", "F", "Br", "CO", "CN"]
    return pd.DataFrame({"smiles": [s + sub for s in scaffolds for sub in subs],
                         "label": [1, 0] * (len(scaffolds) * len(subs) // 2)})


def test_study_compares_the_fixed_and_trained_maps():
    runs = run_trainable_study(sizes=(None,), seeds=(0,), n_qubits=2, max_samples=20,
                               max_iter=3, df=_toy())
    assert set(runs["feature_map"]) == {"fixed", "trained"}
    summary = summarise_trainable(runs)
    assert len(summary) == 2
    assert summary["roc_auc"].between(0, 1).all()


def test_scorecard_marks_wins_ties_and_losses():
    from qudock.scorecard import build_scorecard, verdicts

    runs = pd.DataFrame([{"model": model, "n_train": 100, "seed": seed, "roc_auc": auc}
                         for model, aucs in [("qsvc_pls", [0.80, 0.81, 0.79]),
                                             ("rf_fingerprint", [0.70, 0.71, 0.69])]
                         for seed, auc in enumerate(aucs)])
    table = build_scorecard(runs)
    assert len(table) == 1
    assert table.loc[0, "quantum_model"] == "qsvc_pls"
    assert round(table.loc[0, "delta"], 3) == 0.1
    assert verdicts(table).iloc[0] == "quantum wins"


def test_scorecard_needs_both_sides_of_the_comparison():
    from qudock.scorecard import build_scorecard

    runs = pd.DataFrame([{"model": "qsvc_pls", "n_train": 100, "seed": 0, "roc_auc": 0.8}])
    assert build_scorecard(runs).empty
