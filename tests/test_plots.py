import pandas as pd

from qudock.plots import learning_curve


def test_learning_curve_writes_a_png(tmp_path):
    summary = pd.DataFrame({
        "n_train": [25, 50, 25, 50],
        "model": ["qsvc", "qsvc", "svm_rbf", "svm_rbf"],
        "mean": [0.6, 0.65, 0.62, 0.66],
        "std": [0.02, 0.01, 0.03, 0.02],
    })
    path = tmp_path / "curve.png"
    learning_curve(summary, path)
    assert path.read_bytes()[:4] == b"\x89PNG"
