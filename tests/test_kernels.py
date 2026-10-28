import numpy as np
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import SparsePauliOp, Statevector
from sklearn.metrics import roc_auc_score

from qudock.kernels import KernelSVM, ProjectedKernel, stack, tanimoto_kernel, tune_bandwidth
from qudock.screening import paired_bootstrap, screening_metrics

rng = np.random.default_rng(0)


def test_tanimoto_matches_definition():
    a = np.array([[1, 1, 0, 0], [1, 0, 1, 0]])
    K = tanimoto_kernel(a)
    assert np.allclose(np.diag(K), 1)
    assert np.isclose(K[0, 1], 1 / 3)


def test_tanimoto_of_empty_fingerprints_is_zero():
    assert tanimoto_kernel(np.zeros((2, 3)))[0, 1] == 0


def test_projected_kernel_is_a_valid_gram_matrix():
    X = rng.uniform(0, np.pi, size=(10, 3))
    pk = ProjectedKernel(3, gamma=0.5)
    F = pk.features(X)
    assert F.shape == (10, 9)
    assert np.all(np.abs(F) <= 1 + 1e-9)
    K = pk(X)
    assert np.allclose(np.diag(K), 1)
    assert np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-9


def test_projected_features_are_single_qubit_expectations():
    x = rng.uniform(0, np.pi, size=2)
    state = Statevector(zz_feature_map(2, reps=2).assign_parameters(x))
    expected = [state.expectation_value(SparsePauliOp(label)).real
                for label in ("IX", "IY", "IZ", "XI", "YI", "ZI")]
    assert np.allclose(ProjectedKernel(2).features(x[None])[0], expected)


def test_tune_bandwidth_picks_from_grid():
    X = rng.uniform(0, np.pi, size=(30, 2))
    y = (X[:, 0] > np.pi / 2).astype(int)
    assert tune_bandwidth(X, y, 2, grid=(0.2, 0.5)) in (0.2, 0.5)


def _toy():
    angles = rng.uniform(0, np.pi, size=(60, 2))
    fps = (rng.random((60, 16)) > 0.7).astype(float)
    y = (angles[:, 0] > np.pi / 2).astype(int)
    fps[y == 1, :4] = 1  # make the fingerprint informative too
    return stack(angles, fps), y


def test_every_kernel_kind_learns_a_separable_toy_problem():
    X, y = _toy()
    for kind in ("quantum", "projected", "tanimoto", "hybrid", "hybrid_projected"):
        model = KernelSVM(kind=kind, n_angles=2, C=10, bandwidth=1.0, gamma=5.0)
        model.fit(X[:40], y[:40])
        assert roc_auc_score(y[40:], model.decision_function(X[40:])) > 0.7, kind


def test_hybrid_weights_interpolate_between_kernels():
    X, _ = _toy()
    q = KernelSVM(kind="quantum", n_angles=2).fit(X, np.arange(60) % 2)
    t = KernelSVM(kind="tanimoto", n_angles=2).fit(X, np.arange(60) % 2)
    h = KernelSVM(kind="hybrid", n_angles=2, weight=0.3).fit(X, np.arange(60) % 2)
    assert np.allclose(h._gram(X), 0.3 * q._gram(X) + 0.7 * t._gram(X))


def test_screening_metrics_for_a_perfect_ranking():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    m = screening_metrics(y, -np.arange(10))
    assert m["roc_auc"] == 1 and m["pr_auc"] == 1
    assert m["ef_5"] > 1 and m["bedroc"] > 0.9


def test_paired_bootstrap_detects_a_clear_difference():
    y = np.tile([0, 1], 50)
    good = y + rng.normal(0, 0.3, 100)
    bad = rng.normal(0, 1, 100)
    out = paired_bootstrap(y, good, bad, n_boot=300)
    assert out["delta"] > 0.3 and out["ci_low"] > 0 and out["p_value"] < 0.05


def test_scaling_study_reports_spread_and_accuracy():
    import pandas as pd

    from qudock.scaling import off_diagonal_spread, run_scaling_study

    assert off_diagonal_spread(np.eye(3)) == 0
    scaffolds = ["c1ccccc1", "C1CCNCC1", "c1ccncc1", "C1CCOC1"]
    subs = ["O", "N", "C", "Cl", "F", "Br"]
    toy = pd.DataFrame({"smiles": [s + sub for s in scaffolds for sub in subs],
                        "label": [1, 0] * (len(scaffolds) * len(subs) // 2)})
    out = run_scaling_study(qubits=(2, 3), n_train=18, df=toy)
    assert len(out) == 6
    assert out["roc_auc"].between(0, 1).all()
    assert out["off_diagonal_spread"].gt(0).all()


def test_nystroem_is_exact_with_every_point_as_landmark():
    from qudock.kernels import Nystroem
    from qudock.quantum import ZZKernel

    X = rng.uniform(0, np.pi, size=(30, 3))
    kernel = ZZKernel(3, bandwidth=0.5)
    F = Nystroem(kernel, n_landmarks=30).fit_transform(X)
    assert np.allclose(F @ F.T, kernel(X), atol=1e-6)


def test_nystroem_error_shrinks_with_more_landmarks():
    from qudock.kernels import Nystroem
    from qudock.quantum import ZZKernel

    X = rng.uniform(0, np.pi, size=(120, 4))
    y = (X[:, 0] > np.pi / 2).astype(int)
    kernel = ZZKernel(4, bandwidth=0.4)
    K = kernel(X)
    errors = [np.linalg.norm(Nystroem(kernel, m).fit(X, y).transform(X) @
                             Nystroem(kernel, m).fit(X, y).transform(X).T - K) / np.linalg.norm(K)
              for m in (10, 40, 120)]
    assert errors[0] > errors[1] > errors[2] and errors[2] < 1e-6


def test_nystroem_svm_learns_a_separable_problem():
    from qudock.kernels import NystroemSVM

    local = np.random.default_rng(0)
    X = local.uniform(0, np.pi, size=(160, 3))
    y = (X[:, 0] > np.pi / 2).astype(int)
    for kind in ("quantum", "projected"):
        approx = NystroemSVM(kind=kind, n_landmarks=40, C=1).fit(X[:120], y[:120])
        exact = KernelSVM(kind=kind, n_angles=3, C=1).fit(X[:120], y[:120])
        auc = roc_auc_score(y[120:], approx.decision_function(X[120:]))
        assert auc > 0.8, kind
        assert abs(auc - roc_auc_score(y[120:], exact.decision_function(X[120:]))) < 0.05, kind
