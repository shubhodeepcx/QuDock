import numpy as np
import pandas as pd

from qudock.sweep import kernel_diagnostics, qubit_sweep

SUBS = ["O", "N", "C", "Cl", "F", "Br", "CO", "CN", "CC", "OC"]
TOY = pd.DataFrame({
    "smiles": ["c1ccccc1" + s for s in SUBS] + ["C1CCNCC1" + s for s in SUBS],
    "label": [1, 0] * 10,
})


def test_kernel_diagnostics_has_one_row_per_bandwidth():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, np.pi, size=(20, 3))
    diag = kernel_diagnostics(X, rng.integers(0, 2, 20), bandwidths=(0.5, 1.0))
    assert list(diag["bandwidth"]) == [0.5, 1.0]
    assert (diag["geometric_difference"] > 0).all()


def test_qubit_sweep_on_toy_data():
    perf, diag = qubit_sweep(qubits=[2, 3], seeds=[0], n_kernel=10, df=TOY)
    assert set(perf["n_qubits"]) == {2, 3}
    assert set(diag["n_qubits"]) == {2, 3}
