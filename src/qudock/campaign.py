"""A screening campaign, the way a lab would actually run one.

The benchmarks elsewhere hand a model a fixed training set and score it once.
Real screening is a loop: test a few compounds, retrain on what came back, pick
the next few, repeat. The budget is the number of compounds you can afford to
assay, and the thing you care about is how many inhibitors you have found by
the time the budget runs out.

That loop is also where a quantum kernel has its best case. Its claimed edge is
at small sample sizes, and this is the setting where small sample sizes are not
an artificial restriction but the whole problem: at round one the model has
seen fifty molecules.

Three ways to spend each batch are compared on the same budget and the same
starting molecules:

* ``greedy``: test what the model thinks is most active, which is what a
  screening team wants this quarter,
* ``uncertainty``: test what it is least sure about, which teaches the model
  most for next quarter,
* ``random``: the control, which is what screening without a model looks like.

Labels come from the BACE measurements, so no new chemistry is invented: the
campaign only reveals data it is allowed to see at that round. Molecules of the
held-out scaffold test set are never in the pool, so the ROC-AUC reported each
round is still measured on chemistry the model has never seen.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from qudock.data import load_bace, scaffold_split
from qudock.features import FeaturePipeline, morgan_fingerprints
from qudock.kernels import KernelSVM, stack, tune_bandwidth
from qudock.quantum import _STATE_CACHE

log = logging.getLogger(__name__)

MODELS = ["hybrid_pls", "tanimoto_svm", "rf_fingerprint"]
STRATEGIES = ["greedy", "uncertainty", "random"]


def _fit_and_score(name: str, smiles, y, labelled, pool, test_smiles, n_qubits: int, seed: int):
    """Train on the labelled molecules, then score the pool and the test set."""
    fp = morgan_fingerprints(np.concatenate([smiles[labelled], smiles[pool], test_smiles]))
    n_lab, n_pool = len(labelled), len(pool)
    fp_lab, fp_pool, fp_te = fp[:n_lab], fp[n_lab:n_lab + n_pool], fp[n_lab + n_pool:]

    if name == "rf_fingerprint":
        model = RandomForestClassifier(n_estimators=300, random_state=seed,
                                       n_jobs=-1).fit(fp_lab, y[labelled])
        return (model.predict_proba(fp_pool)[:, 1] - 0.5,
                model.predict_proba(fp_te)[:, 1] - 0.5)

    if name == "tanimoto_svm":
        model = KernelSVM(kind="tanimoto", n_angles=0, C=1).fit(fp_lab, y[labelled])
        return model.decision_function(fp_pool), model.decision_function(fp_te)

    pipe = FeaturePipeline(n_qubits, seed=seed, method="pls", use_descriptors=True)
    angles_lab = pipe.fit_transform(smiles[labelled], y[labelled])
    bandwidth = tune_bandwidth(angles_lab, y[labelled], n_qubits, seed=seed)
    model = KernelSVM(kind="hybrid", n_angles=n_qubits, C=1, weight=0.25,
                      bandwidth=bandwidth).fit(stack(angles_lab, fp_lab), y[labelled])
    out = (model.decision_function(stack(pipe.transform(smiles[pool]), fp_pool)),
           model.decision_function(stack(pipe.transform(test_smiles), fp_te)))
    _STATE_CACHE.clear()        # each round has its own compression, so nothing is reusable
    return out


def _choose(strategy: str, pool_scores: np.ndarray, batch: int, rng) -> np.ndarray:
    """Positions in the pool to test next."""
    if strategy == "random":
        return rng.choice(len(pool_scores), size=min(batch, len(pool_scores)), replace=False)
    order = np.argsort(-pool_scores) if strategy == "greedy" else np.argsort(np.abs(pool_scores))
    return order[:batch]


def run_campaign(model: str, strategy: str, smiles, y, pool: np.ndarray, test_idx: np.ndarray,
                 start: int = 50, batch: int = 25, rounds: int = 8, n_qubits: int = 6,
                 seed: int = 0) -> list[dict]:
    """One campaign: a starting set, then ``rounds`` batches chosen by the model."""
    rng = np.random.default_rng(seed)
    pool = pool.copy()
    labelled = rng.choice(pool, size=start, replace=False)
    pool = np.setdiff1d(pool, labelled)
    test_smiles, y_te = smiles[test_idx], y[test_idx]

    rows = []
    for round_number in range(rounds + 1):
        began = time.perf_counter()
        pool_scores, test_scores = _fit_and_score(model, smiles, y, labelled, pool, test_smiles,
                                                  n_qubits, seed)
        rows.append({"model": model, "strategy": strategy, "seed": seed,
                     "round": round_number, "n_tested": len(labelled),
                     "actives_found": int(y[labelled].sum()),
                     "hit_rate": float(y[labelled].mean()),
                     "roc_auc": float(roc_auc_score(y_te, test_scores)),
                     "seconds": time.perf_counter() - began})
        log.info("%s/%s seed=%d round %d: %d tested, %d actives, auc %.3f", model, strategy,
                 seed, round_number, len(labelled), rows[-1]["actives_found"], rows[-1]["roc_auc"])
        if round_number == rounds or not len(pool):
            break
        picked = _choose(strategy, pool_scores, batch, rng)
        labelled = np.concatenate([labelled, pool[picked]])
        pool = np.delete(pool, picked)
    return rows


def run_campaign_study(models=MODELS, strategies=STRATEGIES, seeds=(0, 1, 2), start: int = 50,
                       batch: int = 25, rounds: int = 8, n_qubits: int = 6,
                       df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = load_bace() if df is None else df
    split = scaffold_split(df["smiles"])
    smiles, y = df["smiles"].values, df["label"].values
    rows = []
    for model in models:
        for strategy in strategies:
            for seed in seeds:
                rows += run_campaign(model, strategy, smiles, y, split.train, split.test,
                                     start=start, batch=batch, rounds=rounds,
                                     n_qubits=n_qubits, seed=seed)
    return pd.DataFrame(rows)


def summarise_campaign(results: pd.DataFrame) -> pd.DataFrame:
    """Mean over seeds of what was found and what was learned, per round."""
    g = results.groupby(["model", "strategy", "round"])
    out = g[["n_tested", "actives_found", "hit_rate", "roc_auc"]].mean()
    out["actives_ci"] = 1.96 * g["actives_found"].sem().fillna(0)
    out["roc_auc_ci"] = 1.96 * g["roc_auc"].sem().fillna(0)
    return out.reset_index()


def campaign_verdict(results: pd.DataFrame) -> pd.DataFrame:
    """At the end of the budget: actives found, and what each strategy cost or bought."""
    last = results[results["round"] == results["round"].max()]
    g = last.groupby(["model", "strategy"])
    out = pd.DataFrame({"n_tested": g["n_tested"].mean(),
                        "actives_found": g["actives_found"].mean(),
                        "actives_sd": g["actives_found"].std(ddof=1).fillna(0),
                        "hit_rate": g["hit_rate"].mean(),
                        "roc_auc": g["roc_auc"].mean()}).reset_index()
    random_hits = out[out["strategy"] == "random"].set_index("model")["actives_found"]
    out["actives_over_random"] = out["actives_found"] - out["model"].map(random_hits)
    return out.sort_values("actives_found", ascending=False).reset_index(drop=True)
