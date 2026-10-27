import numpy as np
import pandas as pd
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeTorino

from qudock import hardware
from qudock.hardware import (
    HardwareConfig,
    counts_to_kernel,
    kernel_circuits,
    pick_molecules,
    reanalyse,
    run_hardware_study,
    save_hardware_study,
)

SUBS = ["O", "N", "C", "Cl", "F", "Br", "CO", "CN", "CC", "OC"]
TOY = pd.DataFrame({
    "smiles": ["c1ccccc1" + s for s in SUBS] + ["C1CCNCC1" + s for s in SUBS],
    "label": [1, 0] * 10,
})


def test_pick_molecules_is_balanced():
    y = np.array([1] * 10 + [0] * 30)
    idx = pick_molecules(y, 8, seed=0)
    assert len(set(idx)) == 8 and y[idx].sum() == 4


def test_counts_to_kernel_fills_both_triangles():
    _, pairs = kernel_circuits(np.zeros((3, 2)), reps=1, bandwidth=1.0)
    K = counts_to_kernel([{"00": 3, "01": 1}, {"00": 1, "11": 1}, {"10": 2}], pairs, 3, 2)
    assert np.allclose(K, [[1, 0.75, 0.5], [0.75, 1, 0], [0.5, 0, 1]])


class _Job:
    def job_id(self):
        return "test-job"

    def usage(self):
        return 0


def _ideal_run(backend, circuits, shots, max_execution_time, seed):
    # stands in for the device: ideal counts, so the test needs no runtime service
    sim = AerSimulator(seed_simulator=seed)
    result = sim.run(transpile(circuits, sim), shots=shots).result()
    return [result.get_counts(i) for i in range(len(circuits))], _Job(), 0


def test_hardware_study_and_reanalyse(tmp_path, monkeypatch):
    monkeypatch.setattr(hardware, "run_on_backend", _ideal_run)
    cfg = HardwareConfig(n_qubits=2, n_molecules=4, shots=256, max_execution_time=None)
    rows, kernels, info = run_hardware_study(cfg, backend=FakeTorino(), df=TOY)
    assert info["n_circuits"] == 6
    assert len(rows) == 3 and rows["rel_frobenius"].between(0, 1).all()
    save_hardware_study(rows, kernels, info, tmp_path)
    again, _, _ = reanalyse(tmp_path, cfg)
    assert np.allclose(again["rel_frobenius"], rows["rel_frobenius"])


def test_hardware_svm_analysis_from_saved_features():
    from qudock.hardware_svm import HardwareSVMConfig, analyse, exact_features

    rng = np.random.default_rng(0)
    cfg = HardwareSVMConfig(n_qubits=2)
    X = rng.uniform(0, np.pi, size=(40, 2))
    y = (X[:, 0] > np.pi / 2).astype(int)
    F = exact_features(X, cfg)
    saved = {"angles": X, "labels": y, "n_train": 30, "exact_features": F,
             "device_features": F + rng.normal(0, 0.02, F.shape)}
    rows = analyse(saved, cfg)
    assert list(rows["features"]) == ["exact simulation", "device",
                                      "trained exact, tested on device"]
    assert rows.loc[0, "feature_error"] == 0
    assert 0 < rows.loc[1, "feature_error"] < 0.05
    assert rows["roc_auc"].between(0, 1).all()
