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
