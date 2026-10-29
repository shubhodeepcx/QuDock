import numpy as np
import pandas as pd

from qudock.campaign import (
    _choose,
    campaign_verdict,
    run_campaign,
    summarise_campaign,
)

SCAFFOLDS = ["c1ccccc1", "C1CCNCC1", "c1ccncc1", "C1CCOC1", "c1ccc2ccccc2c1", "C1CCCCC1",
             "c1ccsc1", "C1CCCC1"]
SUBS = ["O", "N", "C", "Cl", "F", "Br", "CO", "CN", "CCO", "CCN"]


def _toy():
    smiles = [s + sub for s in SCAFFOLDS for sub in SUBS]
    # molecules carrying a halogen are the actives, so there is something to learn
    labels = [int(any(h in sub for h in ("Cl", "F", "Br"))) for _ in SCAFFOLDS for sub in SUBS]
    return pd.DataFrame({"smiles": smiles, "label": labels})


def test_greedy_takes_the_highest_scores_and_uncertainty_the_middle_ones():
    scores = np.array([-2.0, -0.1, 0.2, 3.0])
    rng = np.random.default_rng(0)
    assert set(_choose("greedy", scores, 2, rng)) == {3, 2}
    assert set(_choose("uncertainty", scores, 2, rng)) == {1, 2}
    assert len(_choose("random", scores, 2, rng)) == 2


def test_random_picking_never_asks_for_more_than_the_pool_holds():
    rng = np.random.default_rng(0)
    assert len(_choose("random", np.zeros(3), 10, rng)) == 3


def test_a_campaign_tests_more_molecules_every_round_and_never_repeats_one():
    df = _toy()
    smiles, y = df["smiles"].values, df["label"].values
    pool, test_idx = np.arange(0, 60), np.arange(60, 80)
    rows = run_campaign("tanimoto_svm", "greedy", smiles, y, pool, test_idx,
                        start=20, batch=10, rounds=2, seed=0)
    tested = [r["n_tested"] for r in rows]
    assert tested == [20, 30, 40]
    assert all(r["actives_found"] <= r["n_tested"] for r in rows)
    assert rows[-1]["actives_found"] >= rows[0]["actives_found"]


def test_greedy_finds_more_actives_than_random_on_a_learnable_problem():
    df = _toy()
    smiles, y = df["smiles"].values, df["label"].values
    pool, test_idx = np.arange(0, 60), np.arange(60, 80)
    out = {}
    for strategy in ("greedy", "random"):
        rows = run_campaign("tanimoto_svm", strategy, smiles, y, pool, test_idx,
                            start=20, batch=10, rounds=2, seed=0)
        out[strategy] = rows[-1]["actives_found"]
    assert out["greedy"] > out["random"]


def test_the_summary_and_the_verdict_line_up_with_the_runs():
    rows = pd.DataFrame([
        {"model": "m", "strategy": "greedy", "seed": s, "round": r, "n_tested": 20 + 10 * r,
         "actives_found": 5 + r + s, "hit_rate": 0.3, "roc_auc": 0.7 + 0.01 * r}
        for s in (0, 1) for r in (0, 1, 2)]
        + [{"model": "m", "strategy": "random", "seed": s, "round": r, "n_tested": 20 + 10 * r,
            "actives_found": 3 + r, "hit_rate": 0.2, "roc_auc": 0.65}
           for s in (0, 1) for r in (0, 1, 2)])
    summary = summarise_campaign(rows)
    assert len(summary) == 6
    assert summary["actives_found"].max() == 7.5

    final = campaign_verdict(rows).set_index("strategy")
    assert final.loc["greedy", "actives_found"] == 7.5
    assert final.loc["greedy", "actives_over_random"] == 2.5
    assert final.loc["random", "actives_over_random"] == 0


def test_campaign_study_runs_end_to_end_on_toy_data():
    from qudock.campaign import run_campaign_study

    out = run_campaign_study(models=["tanimoto_svm"], strategies=["greedy"], seeds=(0,),
                             start=20, batch=10, rounds=1, n_qubits=2, df=_toy())
    assert set(out["round"]) == {0, 1}
    assert out["roc_auc"].between(0, 1).all()


def test_mixed_takes_half_the_batch_from_each_end():
    scores = np.array([-2.0, -0.1, 0.2, 3.0, 1.5, -0.3])
    picked = _choose("mixed", scores, 4, np.random.default_rng(0))
    assert len(set(picked)) == 4
    assert {3, 4} <= set(picked)                 # the two highest scores
    assert {1, 2} <= set(picked)                 # the two least certain


def test_mixed_never_picks_the_same_molecule_twice():
    scores = np.array([0.01, -0.02, 0.03, 0.9])
    picked = _choose("mixed", scores, 4, np.random.default_rng(0))
    assert sorted(picked) == [0, 1, 2, 3]


def test_mixed_lands_between_greedy_and_uncertainty_on_a_learnable_problem():
    df = _toy()
    smiles, y = df["smiles"].values, df["label"].values
    pool, test_idx = np.arange(0, 60), np.arange(60, 80)
    found = {}
    for strategy in ("greedy", "mixed", "uncertainty"):
        rows = run_campaign("tanimoto_svm", strategy, smiles, y, pool, test_idx,
                            start=20, batch=10, rounds=2, seed=0)
        found[strategy] = rows[-1]["actives_found"]
    assert found["greedy"] >= found["mixed"] >= found["uncertainty"]
