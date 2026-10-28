"""Rank a library of molecules the way an Alzheimer's project would.

A BACE-1 hit is only useful if it can also reach the brain and look like a
drug, so every molecule gets four independent readouts:

* **activity**: the hybrid quantum + Tanimoto kernel SVM trained on BACE,
  wrapped in conformal prediction so its output is a calibrated p-value for
  "active", plus the prediction set at the chosen confidence;
* **brain penetration**: a Random Forest on fingerprint + descriptors trained
  on the BBBP dataset, giving the probability of crossing the blood-brain barrier;
* **drug-likeness**: RDKit's QED score, and whether the molecule trips a PAINS
  filter (substructures that fool assays);
* **applicability domain**: Tanimoto similarity to the closest BACE training
  molecule, so predictions far from anything seen are marked as such.

The final rank is a weighted geometric mean of activity, brain penetration and
drug-likeness, so a molecule has to be decent on all three to rise; PAINS hits
and molecules outside the domain are flagged rather than silently dropped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from sklearn.ensemble import RandomForestClassifier

from qudock.conformal import ConformalClassifier, describe_set, nearest_neighbour_similarity
from qudock.data import canonical_smiles, scaffold_split
from qudock.features import FeaturePipeline, descriptors, morgan_fingerprints
from qudock.kernels import KernelSVM, stack, tune_bandwidth

log = logging.getLogger(__name__)


@dataclass
class ScreenConfig:
    n_qubits: int = 6
    weight: float = 0.25                 # quantum share of the hybrid kernel
    C: float = 1.0
    confidence: float = 0.8
    domain_threshold: float = 0.35       # nearest-neighbour Tanimoto below this is "far"
    weights: dict[str, float] = field(
        default_factory=lambda: {"activity": 2.0, "bbb": 1.0, "qed": 1.0})
    seed: int = 0


def read_library(path: Path | str) -> pd.DataFrame:
    """SMILES from a CSV (a ``smiles`` column, any case) or a .smi/.txt file, one per line.

    Returns columns ``input`` (as given), ``smiles`` (canonical, or None if RDKit
    could not parse it) and ``name`` when the file had one.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        if "smiles" not in cols:
            raise ValueError(f"{path} has no 'smiles' column")
        out = pd.DataFrame({"input": df[cols["smiles"]].astype(str)})
        if "name" in cols:
            out["name"] = df[cols["name"]].astype(str)
    else:
        rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
        out = pd.DataFrame({"input": [r[0] for r in rows]})
        if any(len(r) > 1 for r in rows):
            out["name"] = [" ".join(r[1:]) for r in rows]
    out["smiles"] = out["input"].map(canonical_smiles)
    return out


def _pains_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    return FilterCatalog(params)


def drug_likeness(smiles) -> pd.DataFrame:
    """QED and the first PAINS alert (empty when clean) for each molecule."""
    catalog = _pains_catalog()
    rows = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        match = catalog.GetFirstMatch(mol)
        rows.append({"qed": QED.qed(mol), "pains": match.GetDescription() if match else ""})
    return pd.DataFrame(rows)


class Screener:
    """Activity, brain penetration, drug-likeness and domain for any list of SMILES."""

    def __init__(self, cfg: ScreenConfig | None = None):
        self.cfg = cfg or ScreenConfig()

    def fit(self, bace: pd.DataFrame, bbbp: pd.DataFrame) -> Screener:
        cfg = self.cfg
        smiles, y = bace["smiles"].values, bace["label"].values
        self.pipe_ = FeaturePipeline(cfg.n_qubits, seed=cfg.seed, method="pls",
                                     use_descriptors=True).fit(smiles, y)
        angles = self.pipe_.transform(smiles)
        self.fp_train_ = morgan_fingerprints(smiles)
        bandwidth = tune_bandwidth(angles, y, cfg.n_qubits, seed=cfg.seed)
        model = KernelSVM(kind="hybrid", n_angles=cfg.n_qubits, C=cfg.C, weight=cfg.weight,
                          bandwidth=bandwidth, seed=cfg.seed)
        # calibrate on the rarest training scaffolds, like the test molecules it will meet
        self.activity_ = ConformalClassifier(model, seed=cfg.seed).fit(
            stack(angles, self.fp_train_), y, smiles=smiles)

        b_smiles = bbbp["smiles"].values
        self.bbb_ = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                           random_state=cfg.seed)
        self.bbb_.fit(self._bbb_features(b_smiles), bbbp["label"].values)
        log.info("screener trained on %d BACE and %d BBBP molecules", len(smiles), len(b_smiles))
        return self

    @staticmethod
    def _bbb_features(smiles) -> np.ndarray:
        return np.hstack([morgan_fingerprints(smiles), descriptors(smiles)])

    def score(self, smiles) -> pd.DataFrame:
        """One row per molecule with every readout and the combined score (not yet sorted)."""
        cfg = self.cfg
        smiles = list(smiles)
        fp = morgan_fingerprints(smiles)
        X = stack(self.pipe_.transform(smiles), fp)
        p = self.activity_.p_values(X)
        sets = p > 1 - cfg.confidence
        out = pd.DataFrame({
            "smiles": smiles,
            "activity_score": self.activity_.decision_function(X),
            "p_active": p[:, 1],
            "p_inactive": p[:, 0],
            "activity_call": [describe_set(s) for s in sets],
            "bbb_probability": self.bbb_.predict_proba(self._bbb_features(smiles))[:, 1],
        })
        out = pd.concat([out, drug_likeness(smiles)], axis=1)
        out["nn_similarity"] = nearest_neighbour_similarity(fp, self.fp_train_)
        out["in_domain"] = out["nn_similarity"] >= cfg.domain_threshold
        out["combined"] = combine(out, cfg.weights)
        return out


def combine(table: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    """Weighted geometric mean of activity p-value, BBB probability and QED."""
    columns = {"activity": "p_active", "bbb": "bbb_probability", "qed": "qed"}
    total = sum(weights.values())
    logs = sum(w * np.log(np.clip(table[columns[k]].to_numpy(float), 1e-6, 1))
               for k, w in weights.items())
    return np.exp(logs / total)


def screen_library(screener: Screener, library: pd.DataFrame) -> pd.DataFrame:
    """Score every parsable molecule, sort best first, and keep unparsable ones at the end."""
    ok = library["smiles"].notna()
    scored = screener.score(library.loc[ok, "smiles"])
    extra = library.loc[ok].drop(columns="smiles").reset_index(drop=True)
    ranked = pd.concat([extra, scored], axis=1)
    ranked = ranked.sort_values(["combined"], ascending=False, kind="stable")
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    failed = library.loc[~ok].drop(columns="smiles").assign(error="could not parse SMILES")
    if len(failed):
        log.warning("skipped %d molecules RDKit could not parse", len(failed))
    return pd.concat([ranked, failed], ignore_index=True)


def holdout_check(bace: pd.DataFrame, bbbp: pd.DataFrame, cfg: ScreenConfig | None = None):
    """Train on the BACE scaffold-split training set and rank its test set.

    Returns (ranked test molecules with their labels, enrichment summary), so
    the ranking can be judged on molecules whose answer is known.
    """
    from qudock.screening import screening_metrics

    split = scaffold_split(bace["smiles"])
    screener = Screener(cfg).fit(bace.iloc[split.train], bbbp)
    test = bace.iloc[split.test].reset_index(drop=True)
    table = screener.score(test["smiles"]).assign(label=test["label"].values)
    summary = {name: screening_metrics(table["label"], table[col])
               for name, col in [("activity only", "activity_score"),
                                 ("combined rank", "combined")]}
    return table, pd.DataFrame(summary).T
