import numpy as np
import pytest

from qudock.features import FeaturePipeline, morgan_fingerprints

TRAIN = ["c1ccccc1O", "c1ccccc1N", "CCO", "CCN", "C1CCCCC1", "c1ccncc1", "CC(=O)O", "CCCC"]
TEST = ["c1ccccc1Cl", "CCCl", "OCCO"]


def test_fingerprints_shape_and_bits():
    fps = morgan_fingerprints(TRAIN, n_bits=256)
    assert fps.shape == (len(TRAIN), 256)
    assert set(np.unique(fps)) <= {0.0, 1.0}


def test_bad_smiles_raises():
    with pytest.raises(ValueError):
        morgan_fingerprints(["nope"])


@pytest.mark.parametrize("n_qubits", [2, 3, 4])
def test_pipeline_outputs_angles_in_range(n_qubits):
    pipe = FeaturePipeline(n_qubits=n_qubits, n_bits=256).fit(TRAIN)
    for x in (pipe.transform(TRAIN), pipe.transform(TEST)):
        assert x.shape[1] == n_qubits
        assert x.min() >= 0 and x.max() <= np.pi
    assert 0 < pipe.explained_variance <= 1


def test_pipeline_is_fit_on_train_only():
    pipe = FeaturePipeline(n_qubits=2, n_bits=256).fit(TRAIN)
    before = pipe.pca_.components_.copy()
    pipe.transform(TEST)
    assert np.array_equal(before, pipe.pca_.components_)


def test_pls_compression_uses_labels_and_stays_in_range():
    smiles = ["CCO", "CCN", "c1ccccc1O", "CC(=O)O", "CCCCCC", "c1ccncc1", "CCOC", "OCCO"]
    y = [0, 1, 0, 1, 0, 1, 0, 1]
    pipe = FeaturePipeline(n_qubits=2, method="pls", use_descriptors=True)
    X = pipe.fit_transform(smiles, y)
    assert X.shape == (8, 2)
    assert X.min() >= 0 and X.max() <= np.pi


def test_pls_without_labels_is_refused():
    with pytest.raises(ValueError):
        FeaturePipeline(n_qubits=2, method="pls").fit(["CCO", "CCN", "CCC"])


def test_descriptors_reject_bad_smiles():
    from qudock.features import descriptors

    with pytest.raises(ValueError, match="nope"):
        descriptors(["CCO", "nope"])


@pytest.mark.parametrize("kwargs", [{"method": "tsne"}, {"n_qubits": 0}])
def test_pipeline_rejects_bad_settings(kwargs):
    with pytest.raises(ValueError):
        FeaturePipeline(**kwargs)


def test_explained_variance_is_pca_only():
    pipe = FeaturePipeline(n_qubits=2, method="pls").fit(TRAIN, [0, 1] * 4)
    with pytest.raises(AttributeError, match="PCA"):
        _ = pipe.explained_variance
