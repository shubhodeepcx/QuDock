import numpy as np
import pandas as pd
from qiskit_ibm_runtime.fake_provider import FakeTorino

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


def test_hardware_study_on_fake_device_and_reanalyse(tmp_path):
    cfg = HardwareConfig(n_qubits=2, n_molecules=4, shots=256, max_execution_time=None)
    rows, kernels, info = run_hardware_study(cfg, backend=FakeTorino(), df=TOY)
    assert info["n_circuits"] == 6
    assert len(rows) == 3 and rows["rel_frobenius"].between(0, 1).all()
    save_hardware_study(rows, kernels, info, tmp_path)
    again, _, _ = reanalyse(tmp_path, cfg)
    assert np.allclose(again["rel_frobenius"], rows["rel_frobenius"])
