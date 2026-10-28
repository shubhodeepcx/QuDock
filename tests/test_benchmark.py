import numpy as np
import pandas as pd

from qudock.benchmark import (
    BenchmarkConfig,
    bootstrap_ci,
    compare_to_best_classical,
    run_benchmark,
    subsample,
    summarise,
)

# two scaffold families, each with both labels, so the scaffold split
# leaves both classes on each side
SUBS = ["O", "N", "C", "Cl", "F", "Br", "CO", "CN", "CC", "OC"]
TOY = pd.DataFrame({
    "smiles": ["c1ccccc1" + s for s in SUBS] + ["C1CCNCC1" + s for s in SUBS],
    "label": [1, 0] * 10,
})


def test_subsample_is_stratified_and_seeded():
    y = np.array([0] * 30 + [1] * 30)
    a, b = subsample(y, 20, seed=3), subsample(y, 20, seed=3)
    assert (a == b).all() and len(a) == 20 and y[a].sum() == 10
    assert len(subsample(y, None, seed=0)) == 60


def test_benchmark_runs_end_to_end_on_toy_data():
    cfg = BenchmarkConfig(n_qubits=2, sizes=[None], seeds=[0, 1], test_frac=0.3)
    results = run_benchmark(cfg, df=TOY)
    assert set(results["model"]) == {"svm_rbf", "random_forest", "logreg", "qsvc"}
    assert len(results) == 8
    assert results["roc_auc"].between(0, 1).all()
    assert set(summarise(results).columns) == {"n_train", "model", "mean", "std"}
    assert len(compare_to_best_classical(results)) == 1


def test_bootstrap_ci_brackets_the_auc():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    score = y + rng.normal(0, 1, 200)
    lo, hi = bootstrap_ci(y, score, n_boot=200)
    assert 0.5 < lo < hi < 1


def test_final_study_runs_end_to_end_on_toy_data():
    from qudock.final_study import (
        MODELS,
        FinalConfig,
        compare_pairs,
        run_final_study,
        summarise_final,
    )

    cfg = FinalConfig(n_qubits=2, sizes=[None], seeds=[0], test_frac=0.3)
    results, test_scores = run_final_study(cfg, df=TOY)
    assert set(results["model"]) == set(MODELS)
    assert results["roc_auc"].between(0, 1).all()
    assert results.loc[results["model"] == "hybrid_pls", "weight"].notna().all()
    assert set(test_scores.columns) == {"label", *MODELS}
    assert {"roc_auc", "roc_auc_ci", "ef_1", "bedroc"} <= set(summarise_final(results).columns)
    pairs = compare_pairs(test_scores, [("hybrid_pls", "tanimoto_svm")])
    assert {"delta", "ci_low", "ci_high", "p_value"} <= set(pairs.columns)


def test_comparison_skips_sizes_without_both_sides():
    only_classical = pd.DataFrame({"model": ["svm_rbf", "logreg"], "n_train": [50, 50],
                                   "seed": [0, 0], "roc_auc": [0.7, 0.6]})
    out = compare_to_best_classical(only_classical)
    assert out.empty and "delta" in out.columns
