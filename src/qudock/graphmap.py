"""A feature map built from the molecule's pharmacophore graph, not from a compressed fingerprint.

The ZZ map on PCA or PLS angles squeezes 2,048 fingerprint bits into six
numbers and then entangles neighbouring qubits in an arbitrary order. Here
each qubit is one pharmacophore family, the groups a binding site actually
recognises:

    donor, acceptor, positive ionizable, negative ionizable, aromatic, hydrophobe

BACE-1 cleaves its substrate with two catalytic aspartates, and most potent
inhibitors place a protonated amidine or amine between them, flanked by
hydrophobic groups for the S1 and S3 pockets, so these are the families that
matter for this target.

* qubit angle x_q: how much of family q the molecule has, pi * c / (c + 2)
  for c features, so it saturates instead of growing without bound;
* coupling w_qr: how close families q and r sit in the molecule, summed over
  every pair of features as exp(-d / 2) for a topological distance of d bonds,
  mapped to pi * s / (s + 1).

The circuit per repetition is H on every qubit, RZ(2 x_q), then a ZZ rotation
of 2 w_qr between every pair of qubits. The entanglement pattern therefore
differs from molecule to molecule and follows its pharmacophore geometry: two
molecules with the same groups arranged differently get different states.
All rotations in a layer are diagonal, so the state is simulated exactly with
numpy, batched over molecules.
"""

from __future__ import annotations

import os
from functools import lru_cache
from itertools import combinations

import numpy as np
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures

from qudock.features import parse
from qudock.trainable import _apply_h

FAMILIES = ("Donor", "Acceptor", "PosIonizable", "NegIonizable", "Aromatic", "Hydrophobe")
N_QUBITS = len(FAMILIES)
PAIRS = list(combinations(range(N_QUBITS), 2))


@lru_cache(maxsize=1)
def _factory():
    return ChemicalFeatures.BuildFeatureFactory(
        os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef"))


@lru_cache(maxsize=20_000)
def _graph_row(smiles: str) -> tuple[float, ...]:
    mol = parse(smiles)
    dist = Chem.GetDistanceMatrix(mol)
    atoms: list[list[tuple[int, ...]]] = [[] for _ in FAMILIES]
    for feat in _factory().GetFeaturesForMol(mol):
        if feat.GetFamily() in FAMILIES:
            atoms[FAMILIES.index(feat.GetFamily())].append(feat.GetAtomIds())
    counts = [len(a) for a in atoms]
    closeness = []
    for q, r in PAIRS:
        s = sum(np.exp(-dist[np.ix_(a, b)].min() / 2) for a in atoms[q] for b in atoms[r])
        closeness.append(float(s))
    return tuple(counts) + tuple(closeness)


def pharmacophore_counts(smiles) -> np.ndarray:
    """(n, 6 + 15) raw counts per family, then closeness per family pair."""
    return np.array([_graph_row(s) for s in smiles], dtype=float).reshape(
        len(smiles), N_QUBITS + len(PAIRS))


def graph_angles(smiles) -> np.ndarray:
    """Qubit angles and pair couplings, both in [0, pi), ready for :class:`GraphKernel`."""
    raw = pharmacophore_counts(smiles)
    counts, closeness = raw[:, :N_QUBITS], raw[:, N_QUBITS:]
    return np.hstack([np.pi * counts / (counts + 2), np.pi * closeness / (closeness + 1)])


@lru_cache(maxsize=4)
def _signs(n_qubits: int) -> tuple[np.ndarray, np.ndarray]:
    bits = np.arange(2**n_qubits)
    z = np.stack([1 - 2 * ((bits >> (n_qubits - 1 - q)) & 1) for q in range(n_qubits)])
    zz = np.stack([z[q] * z[r] for q, r in combinations(range(n_qubits), 2)])
    return z.astype(float), zz.astype(float)


class GraphKernel:
    """Fidelity kernel of the pharmacophore-graph feature map."""

    def __init__(self, reps: int = 2, bandwidth: float = 1.0, n_qubits: int = N_QUBITS):
        self.reps = reps
        self.bandwidth = bandwidth
        self.n_qubits = n_qubits

    def statevectors(self, A: np.ndarray) -> np.ndarray:
        """``A`` is (n, n_qubits + pairs) from :func:`graph_angles`."""
        A = self.bandwidth * np.asarray(A, dtype=float)
        n = self.n_qubits
        x, w = A[:, :n], A[:, n:]
        z, zz = _signs(n)
        state = np.full((len(A), 2**n), 2 ** (-n / 2), dtype=complex)
        for layer in range(self.reps):
            if layer:
                for q in range(n):
                    state = _apply_h(state, q, n)
            state = state * np.exp(-1j * (x @ z + w @ zz))
        return state

    def __call__(self, A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
        sa = self.statevectors(A)
        sb = sa if B is None else self.statevectors(B)
        return np.abs(sa.conj() @ sb.T) ** 2


def run_graph_study(seeds=(0, 1, 2), sizes=(100, 400, None), reps: int = 2, df=None):
    """Graph feature map against the ZZ map and classical models on the same information.

    Every model is tuned by inner CV and tested on the scaffold split:

    * ``graph_quantum``: SVM on the pharmacophore-graph quantum kernel
    * ``graph_rbf``: SVM-RBF on exactly the same 21 graph numbers (fair classical twin)
    * ``graph_rf``: Random Forest on the same 21 numbers
    * ``qsvc_pls``: the ZZ map on 6 PLS angles, from the final study
    """
    import pandas as pd
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.svm import SVC

    from qudock.benchmark import subsample
    from qudock.classical import make_classical, scores
    from qudock.data import load_bace, scaffold_split
    from qudock.features import FeaturePipeline
    from qudock.kernels import KernelSVM, tune_bandwidth
    from qudock.screening import screening_metrics

    df = load_bace() if df is None else df
    split = scaffold_split(df["smiles"])
    smiles, y = df["smiles"].values, df["label"].values
    smi_tr, y_tr, smi_te, y_te = (smiles[split.train], y[split.train],
                                  smiles[split.test], y[split.test])
    G_all, G_te = graph_angles(smi_tr), graph_angles(smi_te)

    rows = []
    for size in sizes:
        for seed in seeds:
            keep = subsample(y_tr, size, seed)
            G, yk = G_all[keep], y_tr[keep]
            cv = StratifiedKFold(5 if len(keep) >= 100 else 3, shuffle=True, random_state=seed)
            outputs = {}
            best = None
            for bw in (0.25, 0.5, 1.0):
                kernel = GraphKernel(reps, bw)
                search = GridSearchCV(SVC(kernel="precomputed", random_state=seed),
                                      {"C": [0.1, 1, 10]}, cv=cv, scoring="roc_auc")
                search.fit(kernel(G), yk)
                if best is None or search.best_score_ > best[0]:
                    best = (search.best_score_, kernel, search.best_estimator_)
            _, kernel, svc = best
            outputs["graph_quantum"] = svc.decision_function(kernel(G_te, G))
            for name, base in (("graph_rbf", "svm_rbf"), ("graph_rf", "random_forest")):
                model = make_classical(base, seed=seed, n_train=len(keep)).fit(G, yk)
                outputs[name] = scores(model, G_te)
            pipe = FeaturePipeline(6, seed=seed, method="pls", use_descriptors=True)
            A, A_te = pipe.fit_transform(smi_tr[keep], yk), pipe.transform(smi_te)
            bw = tune_bandwidth(A, yk, 6, seed=seed)
            qsvc = GridSearchCV(KernelSVM(kind="quantum", n_angles=6, bandwidth=bw, seed=seed),
                                {"C": [0.1, 1, 10]}, cv=cv, scoring="roc_auc").fit(A, yk)
            outputs["qsvc_pls"] = qsvc.decision_function(A_te)
            for name, s in outputs.items():
                rows.append({"model": name, "n_train": len(keep), "seed": seed,
                             **screening_metrics(y_te, s)})
    return pd.DataFrame(rows)
