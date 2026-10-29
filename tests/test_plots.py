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


def test_graph_study_plot(tmp_path):
    import pandas as pd

    from qudock.plots import graph_study

    runs = pd.DataFrame({"model": ["graph_quantum", "graph_rbf"] * 2, "n_train": [100] * 4,
                         "seed": [0, 0, 1, 1], "roc_auc": [0.7, 0.68, 0.72, 0.69]})
    graph_study(runs, tmp_path / "g.png")
    assert (tmp_path / "g.png").stat().st_size > 0


def test_robustness_study_writes_a_png(tmp_path):
    from qudock.plots import robustness_study

    summary = pd.DataFrame({"split_seed": [0, 1], "delta": [-0.05, 0.01]})
    spread = pd.DataFrame({"model": ["qsvc_pls", "rf_fingerprint"], "mean": [0.79, 0.88],
                           "worst": [0.76, 0.86], "best": [0.84, 0.90]})
    path = tmp_path / "robustness.png"
    robustness_study(summary, spread, path)
    assert path.read_bytes()[:4] == b"\x89PNG"


def test_campaign_study_writes_a_png(tmp_path):
    from qudock.plots import campaign_study

    summary = pd.DataFrame({
        "model": ["hybrid_pls"] * 2 + ["rf_fingerprint"] * 2,
        "strategy": ["greedy"] * 4, "round": [0, 1, 0, 1], "n_tested": [50, 75, 50, 75],
        "actives_found": [20, 40, 20, 42], "hit_rate": [0.4, 0.53, 0.4, 0.56],
        "roc_auc": [0.70, 0.72, 0.71, 0.74], "actives_ci": [1.0] * 4,
        "roc_auc_ci": [0.01] * 4})
    path = tmp_path / "campaign.png"
    campaign_study(summary, path)
    assert path.read_bytes()[:4] == b"\x89PNG"


def test_cost_study_writes_a_png(tmp_path):
    from qudock.plots import cost_study

    results = pd.DataFrame({
        "model": ["qsvc_pls"] * 2 + ["rf_fingerprint"] * 2,
        "kind": ["fidelity"] * 2 + ["classical"] * 2, "n_train": [100, 400] * 2,
        "total_seconds": [5.0, 40.0, 0.5, 1.0], "circuits": [35000, 300000, 0, 0],
        "device_seconds": [10600.0, 90900.0, 0.0, 0.0]})
    path = tmp_path / "cost.png"
    cost_study(results, path)
    assert path.read_bytes()[:4] == b"\x89PNG"
