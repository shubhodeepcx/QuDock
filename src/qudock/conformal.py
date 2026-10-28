"""Calibrated confidence for every prediction (Mondrian inductive conformal prediction).

Part of the training set is held back for calibration. For a new molecule we
ask, for each possible label, how unusual its score would be among the
calibration molecules of that label. Labels that are not too unusual form the
prediction set: {active}, {inactive}, both ("don't know") or neither. Within
each class, the true label is left out of the set at most 1 - confidence of
the time, whatever model produced the scores, as long as new molecules look
like the calibration molecules.

Test molecules come from rare, unseen scaffolds, so a random calibration split
promises too much. Passing the training SMILES to ``fit`` holds back the rarest
scaffolds of the training set instead, the same way the test set was made, so
calibration molecules are new chemistry too.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from qudock.classical import scores
from qudock.data import scaffold_split


class ConformalClassifier:
    def __init__(self, model, calib_frac: float = 0.25, seed: int = 0):
        self.model, self.calib_frac, self.seed = model, calib_frac, seed

    def fit(self, X, y, smiles=None):
        X, y = np.asarray(X), np.asarray(y)
        if not 0 < self.calib_frac < 1:
            raise ValueError(f"calib_frac must be between 0 and 1, got {self.calib_frac}")
        if smiles is None:
            fit_idx, cal_idx = train_test_split(np.arange(len(y)), test_size=self.calib_frac,
                                                stratify=y, random_state=self.seed)
        else:
            split = scaffold_split(smiles, test_frac=self.calib_frac)
            fit_idx, cal_idx = split.train, split.test
        self.model_ = clone(self.model).fit(X[fit_idx], y[fit_idx])
        s = self._score(X[cal_idx])
        # nonconformity: low score is strange for an active, high score for an inactive
        self.cal_ = {1: np.sort(-s[y[cal_idx] == 1]), 0: np.sort(s[y[cal_idx] == 0])}
        return self

    def _score(self, X):
        return scores(self.model_, X)

    def p_values(self, X) -> np.ndarray:
        """(n, 2) array: p-value of the labels inactive and active for each molecule."""
        s = self._score(np.asarray(X))
        out = np.empty((len(s), 2))
        for label, alpha in ((0, s), (1, -s)):
            cal = self.cal_[label]
            # share of calibration molecules at least as strange as this one
            n_ge = len(cal) - np.searchsorted(cal, alpha, side="left")
            out[:, label] = (n_ge + 1) / (len(cal) + 1)
        return out

    def predict_set(self, X, confidence: float = 0.8) -> np.ndarray:
        """(n, 2) boolean array: is inactive / active in the prediction set."""
        if not 0 < confidence < 1:
            raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
        return self.p_values(X) > 1 - confidence

    def decision_function(self, X):
        return self._score(np.asarray(X))


def describe_set(in_set) -> str:
    inactive, active = bool(in_set[0]), bool(in_set[1])
    if active and not inactive:
        return "active"
    if inactive and not active:
        return "inactive"
    return "don't know" if active else "unlike anything seen"


def conformal_report(y, sets) -> dict[str, float]:
    """Coverage per class and how often the model commits to a single label."""
    y, sets = np.asarray(y), np.asarray(sets)
    covered = sets[np.arange(len(y)), y]
    single = sets.sum(axis=1) == 1
    return {"coverage": covered.mean(),
            "coverage_active": covered[y == 1].mean(),
            "coverage_inactive": covered[y == 0].mean(),
            "single_label": single.mean(),
            "single_label_accuracy": (sets[single, y[single]]).mean() if single.any()
            else float("nan")}


def nearest_neighbour_similarity(fp_test, fp_train) -> np.ndarray:
    """Tanimoto similarity of each test molecule to its closest training molecule."""
    from qudock.kernels import tanimoto_kernel

    return tanimoto_kernel(fp_test, fp_train).max(axis=1)


def run_confidence_study(n_qubits: int = 6, seed: int = 0, df=None):
    """Prediction sets and accuracy by similarity, for the quantum, hybrid and Tanimoto models."""
    import pandas as pd
    from sklearn.metrics import roc_auc_score

    from qudock.data import load_bace, scaffold_split
    from qudock.features import FeaturePipeline, morgan_fingerprints
    from qudock.kernels import KernelSVM, stack, tune_bandwidth

    df = load_bace() if df is None else df
    split = scaffold_split(df["smiles"])
    smiles, y = df["smiles"].values, df["label"].values
    tr, te = split.train, split.test
    pipe = FeaturePipeline(n_qubits, seed=seed, method="pls", use_descriptors=True)
    pipe.fit(smiles[tr], y[tr])
    fp_tr, fp_te = morgan_fingerprints(smiles[tr]), morgan_fingerprints(smiles[te])
    X_tr = stack(pipe.transform(smiles[tr]), fp_tr)
    X_te = stack(pipe.transform(smiles[te]), fp_te)
    bandwidth = tune_bandwidth(pipe.transform(smiles[tr]), y[tr], n_qubits, seed=seed)

    models = {"qsvc_pls": KernelSVM(kind="quantum", n_angles=n_qubits, C=1, bandwidth=bandwidth),
              "hybrid_pls": KernelSVM(kind="hybrid", n_angles=n_qubits, C=1, weight=0.25,
                                      bandwidth=bandwidth),
              "tanimoto_svm": KernelSVM(kind="tanimoto", n_angles=n_qubits, C=1)}
    similarity = nearest_neighbour_similarity(fp_te, fp_tr)
    edges = np.quantile(similarity, [0, 1 / 3, 2 / 3, 1.0])

    sets_rows, domain_rows = [], []
    for name, model in models.items():
        cc = ConformalClassifier(model, seed=seed).fit(X_tr, y[tr], smiles=smiles[tr])
        for confidence in (0.7, 0.8, 0.9):
            sets_rows.append({"model": name, "confidence": confidence,
                              **conformal_report(y[te], cc.predict_set(X_te, confidence))})
        score = cc.decision_function(X_te)
        for lo, hi in pairwise(edges):
            keep = (similarity >= lo) & (similarity <= hi)
            domain_rows.append({"model": name, "similarity_from": lo, "similarity_to": hi,
                                "n_molecules": int(keep.sum()),
                                "roc_auc": roc_auc_score(y[te][keep], score[keep])})
    return pd.DataFrame(sets_rows), pd.DataFrame(domain_rows)
