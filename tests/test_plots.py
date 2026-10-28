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


def test_explain_study_writes_a_png(tmp_path):
    from qudock.plots import explain_study

    results = pd.DataFrame({
        "top_atoms": [0.8, 0.2, 0.5],
        "random_atoms": [0.3, 0.4, 0.1],
        "gap": [0.5, -0.2, 0.4],
        "top_substructure": ["c1ccccc1", "CC(=O)N", "CCO"],
    })
    path = tmp_path / "explain.png"
    explain_study(results, path)
    assert path.read_bytes()[:4] == b"\x89PNG"
