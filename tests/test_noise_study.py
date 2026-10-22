import pandas as pd

from qudock.noise_study import NoiseConfig, run_noise_study
from qudock.plots import noise_study

SUBS = ["O", "N", "C", "Cl", "F", "Br", "CO", "CN", "CC", "OC"]
TOY = pd.DataFrame({
    "smiles": ["c1ccccc1" + s for s in SUBS] + ["C1CCNCC1" + s for s in SUBS],
    "label": [1, 0] * 10,
})


def test_noise_study_runs_on_toy_data(tmp_path):
    cfg = NoiseConfig(n_qubits=2, n_train=6, n_test=4, shots=[64], reps=[1],
                      backend="FakeTorino", seeds=[0])
    results = run_noise_study(cfg, df=TOY)
    # exact + (shots only, FakeTorino) x (raw, psd repaired)
    assert len(results) == 5
    assert set(results["noise"]) == {"exact", "shots only", "FakeTorino"}
    assert results["roc_auc"].between(0, 1).all()
    noise_study(results, tmp_path / "noise.png")
    assert (tmp_path / "noise.png").exists()
