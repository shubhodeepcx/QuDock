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
    """Mondrian inductive conformal prediction, optionally weighted by similarity.

    With ``similarity_bandwidth`` set and fingerprints passed to ``fit`` and
    ``p_values``, each calibration molecule counts in proportion to
    ``exp(-(1 - tanimoto) / bandwidth)``, its chemical closeness to the molecule
    being predicted. This is weighted conformal prediction (Tibshirani et al.,
    2019) with a similarity proxy for the likelihood ratio, and it is the part
    that matters under a scaffold split: unweighted calibration treats a
    molecule from a scaffold the model has never seen as if it were drawn from
    the same distribution as the calibration set, which is exactly when the
    promised coverage is not delivered.
    """

    def __init__(self, model, calib_frac: float = 0.25, seed: int = 0,
                 similarity_bandwidth: float | None = None):
        self.model, self.calib_frac, self.seed = model, calib_frac, seed
        self.similarity_bandwidth = similarity_bandwidth

    def fit(self, X, y, smiles=None, fingerprints=None):
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
        if fingerprints is not None:
            fp = np.asarray(fingerprints)[cal_idx]
            # kept in the same order as the sorted nonconformity scores above
            self.cal_fp_ = {1: fp[y[cal_idx] == 1][np.argsort(-s[y[cal_idx] == 1])],
                            0: fp[y[cal_idx] == 0][np.argsort(s[y[cal_idx] == 0])]}
        else:
            self.cal_fp_ = None
        return self

    def _score(self, X):
        return scores(self.model_, X)

    def _weights(self, label: int, fingerprints) -> np.ndarray:
        """(n_test, n_calibration) similarity weights; a far molecule counts for little."""
        from qudock.kernels import tanimoto_kernel

        similarity = tanimoto_kernel(np.asarray(fingerprints), self.cal_fp_[label])
        return np.exp(-(1 - similarity) / self.similarity_bandwidth)

    def p_values(self, X, fingerprints=None) -> np.ndarray:
        """(n, 2) array: p-value of the labels inactive and active for each molecule."""
        s = self._score(np.asarray(X))
        weighted = self.similarity_bandwidth is not None and fingerprints is not None
        if weighted and self.cal_fp_ is None:
            raise ValueError("pass fingerprints to fit as well to weight by similarity")
        out = np.empty((len(s), 2))
        for label, alpha in ((0, s), (1, -s)):
            cal = self.cal_[label]
            if not weighted:
                # share of calibration molecules at least as strange as this one
                n_ge = len(cal) - np.searchsorted(cal, alpha, side="left")
                out[:, label] = (n_ge + 1) / (len(cal) + 1)
                continue
            w = self._weights(label, fingerprints)
            strange = cal[None, :] >= alpha[:, None]
            # the molecule itself always counts, with the weight of a perfect match
            out[:, label] = ((w * strange).sum(axis=1) + 1) / (w.sum(axis=1) + 1)
        return out

    def predict_set(self, X, confidence: float = 0.8, fingerprints=None) -> np.ndarray:
        """(n, 2) boolean array: is inactive / active in the prediction set."""
        if not 0 < confidence < 1:
            raise ValueError(f"confidence must be between 0 and 1, got {confidence}")
        return self.p_values(X, fingerprints) > 1 - confidence

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
            "dont_know": (sets.sum(axis=1) == 2).mean(),
            "no_label": (sets.sum(axis=1) == 0).mean(),
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


def run_tradeoff_study(n_qubits: int = 6, seed: int = 0, bandwidths=(None, 0.4),
                       confidences=(0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99), df=None):
    """What a requested confidence really delivers, and what it costs in decisiveness.

    Under a scaffold split the promised coverage is not delivered, so the useful
    question is not "is it calibrated" but "what do I have to ask for, and how
    often does the model then refuse to commit". Both are measured here, for the
    plain calibration and for one weighted by chemical similarity, which is the
    standard correction for covariate shift (Tibshirani et al., 2019).
    """
    import pandas as pd

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

    models = {"hybrid_pls": KernelSVM(kind="hybrid", n_angles=n_qubits, C=1, weight=0.25,
                                      bandwidth=bandwidth),
              "tanimoto_svm": KernelSVM(kind="tanimoto", n_angles=n_qubits, C=1)}
    rows = []
    for name, model in models.items():
        for h in bandwidths:
            cc = ConformalClassifier(model, seed=seed, similarity_bandwidth=h)
            cc.fit(X_tr, y[tr], smiles=smiles[tr], fingerprints=fp_tr)
            for confidence in confidences:
                sets = cc.predict_set(X_te, confidence, fingerprints=fp_te)
                rows.append({"model": name,
                             "calibration": "plain" if h is None else f"similarity h={h}",
                             "confidence": confidence, **conformal_report(y[te], sets)})
    return pd.DataFrame(rows)


def confidence_needed(tradeoff, target: float = 0.9):
    """The confidence each method must be asked for to really deliver ``target``."""
    import pandas as pd

    rows = []
    for (model, calibration), g in tradeoff.groupby(["model", "calibration"]):
        g = g.sort_values("confidence")
        enough = g[g["coverage"] >= target]
        row = {"model": model, "calibration": calibration, "target": target}
        if len(enough):
            first = enough.iloc[0]
            row |= {"ask_for": first["confidence"], "coverage": first["coverage"],
                    "single_label": first["single_label"], "dont_know": first["dont_know"]}
        else:
            best = g.iloc[-1]
            row |= {"ask_for": float("nan"), "coverage": best["coverage"],
                    "single_label": best["single_label"], "dont_know": best["dont_know"]}
        rows.append(row)
    return pd.DataFrame(rows)
