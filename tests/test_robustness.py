import numpy as np
import pandas as pd
import pytest

from qudock.data import load_bace, scaffold_split
from qudock.robustness import (
    paired_split_test,
    per_model_spread,
    summarise_robustness,
    verdict,
)


def _runs():
    rows = []
    for split_seed, (q, c) in enumerate([(0.74, 0.78), (0.79, 0.77), (0.71, 0.80), (0.76, 0.79)]):
        rows += [{"split_seed": split_seed, "model": "qsvc_pls", "roc_auc": q, "seed": 0},
                 {"split_seed": split_seed, "model": "rf_fingerprint", "roc_auc": c, "seed": 0}]
    return pd.DataFrame(rows)


def test_the_same_seed_gives_the_same_split():
    smiles = load_bace()["smiles"]
    a, b = scaffold_split(smiles, seed=3), scaffold_split(smiles, seed=3)
    assert np.array_equal(a.test, b.test)


def test_seeded_splits_differ_from_each_other_but_stay_the_same_size():
    smiles = load_bace()["smiles"]
    fixed = scaffold_split(smiles)
    first, second = scaffold_split(smiles, seed=0), scaffold_split(smiles, seed=1)
    assert len(first.test) == len(fixed.test)
    assert not np.array_equal(first.test, second.test)
    assert not np.array_equal(first.test, fixed.test)


def test_no_molecule_is_on_both_sides_of_a_seeded_split():
    smiles = load_bace()["smiles"]
    split = scaffold_split(smiles, seed=2)
    assert not set(split.train) & set(split.test)
    assert len(split.train) + len(split.test) == len(smiles)


def test_a_seeded_split_keeps_scaffolds_whole():
    from qudock.data import murcko_scaffold

    smiles = load_bace()["smiles"].values
    split = scaffold_split(smiles, seed=1)
    train = {murcko_scaffold(s) for s in smiles[split.train]}
    test = {murcko_scaffold(s) for s in smiles[split.test]}
    assert not train & test


def test_the_summary_names_the_best_model_on_each_side():
    summary = summarise_robustness(_runs())
    assert list(summary["best_quantum"]) == ["qsvc_pls"] * 4
    assert list(summary["best_classical"]) == ["rf_fingerprint"] * 4
    assert summary["delta"].round(2).tolist() == [-0.04, 0.02, -0.09, -0.03]


def test_the_verdict_counts_the_splits_the_quantum_side_won():
    v = verdict(summarise_robustness(_runs()))
    assert v["n_splits"] == 4
    assert v["splits_quantum_ahead"] == 1
    assert v["worst_delta"] < v["mean_delta"] < v["best_delta"]


def test_the_spread_table_shows_the_range_between_splits():
    spread = per_model_spread(_runs()).set_index("model")
    assert spread.loc["qsvc_pls", "range"] == pytest.approx(0.08)
    assert spread.loc["qsvc_pls", "mean"] > 0.7


def test_the_paired_test_compares_split_by_split():
    out = paired_split_test(_runs(), "qsvc_pls", "rf_fingerprint")
    assert out["n_splits"] == 4
    assert out["splits_ahead"] == 1
    assert out["ci_low"] < out["mean_difference"] < out["ci_high"]
    assert paired_split_test(_runs(), "qsvc_pls", "not_a_model") == {}


def test_split_noise_and_seed_noise_are_separated():
    from qudock.robustness import split_versus_seed_noise

    rows = [{"model": "m", "split_seed": s, "seed": d, "roc_auc": 0.70 + 0.02 * s + 0.002 * d}
            for s in range(3) for d in range(2)]
    out = split_versus_seed_noise(pd.DataFrame(rows)).iloc[0]
    assert out["between_splits"] > out["within_split_between_seeds"]
    assert out["within_split_between_seeds"] == pytest.approx(0.002 / np.sqrt(2), abs=1e-6)
