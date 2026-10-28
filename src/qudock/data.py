"""Load, clean and split the MoleculeNet BACE dataset."""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

MOLECULENET = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets"
BACE_URL = f"{MOLECULENET}/bace.csv"
# second task: does the molecule cross the blood-brain barrier, which any
# Alzheimer's drug has to do, so the same pipeline can be checked on it
BBBP_URL = f"{MOLECULENET}/BBBP.csv"
DEFAULT_CACHE = Path("data/cache")

log = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")


def download(url: str, path: Path, timeout: float = 60) -> Path:
    """Download a dataset once and return its local path.

    The file is written next to its final name and renamed at the end, so an
    interrupted download never leaves a half-written CSV in the cache.
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s", url)
        partial = path.with_name(path.name + ".part")
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                partial.write_bytes(response.read())
        except OSError as err:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"could not download {url}: {err}") from err
        partial.replace(path)
    return path


def download_bace(cache_dir: Path | str = DEFAULT_CACHE) -> Path:
    return download(BACE_URL, Path(cache_dir) / "bace.csv")


def canonical_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def clean(df: pd.DataFrame, smiles_col: str = "mol", label_col: str = "Class") -> pd.DataFrame:
    """Canonicalise SMILES, drop unparsable molecules and duplicates.

    Duplicates with the same label keep their first copy. Duplicates with
    conflicting labels are dropped entirely, since we can't tell which is right.
    Returns a frame with columns ``smiles`` and ``label``.
    """
    out = pd.DataFrame({
        "smiles": df[smiles_col].map(canonical_smiles),
        "label": df[label_col].astype(int),
    })
    bad = out["smiles"].isna()
    if bad.any():
        log.info("dropped %d unparsable molecules", int(bad.sum()))
    out = out[~bad]

    n_labels = out.groupby("smiles")["label"].nunique()
    conflicting = n_labels[n_labels > 1].index
    if len(conflicting):
        log.info("dropped %d molecules with conflicting labels", len(conflicting))
    out = out[~out["smiles"].isin(conflicting)]

    n_before = len(out)
    out = out.drop_duplicates("smiles")
    if len(out) < n_before:
        log.info("dropped %d duplicate molecules", n_before - len(out))
    return out.reset_index(drop=True)


def load_bace(cache_dir: Path | str = DEFAULT_CACHE) -> pd.DataFrame:
    """Return the cleaned BACE dataset (columns ``smiles``, ``label``)."""
    return clean(pd.read_csv(download_bace(cache_dir)))


def load_bbbp(cache_dir: Path | str = DEFAULT_CACHE) -> pd.DataFrame:
    """Return the cleaned BBBP dataset (blood-brain barrier penetration)."""
    return clean(pd.read_csv(download(BBBP_URL, Path(cache_dir) / "BBBP.csv")),
                 smiles_col="smiles", label_col="p_np")


DATASETS = {"bace": load_bace, "bbbp": load_bbbp}


def murcko_scaffold(smiles: str) -> str:
    return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)


@dataclass
class Split:
    train: np.ndarray
    test: np.ndarray


def scaffold_split(smiles: list[str] | pd.Series, test_frac: float = 0.2) -> Split:
    """Bemis-Murcko scaffold split, as recommended by MoleculeNet.

    Molecules sharing a scaffold always land on the same side. Scaffold groups
    are assigned largest first to train until it is full, so the test set gets
    the rarer scaffolds, which is the harder and more realistic setting.
    """
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(smiles):
        groups.setdefault(murcko_scaffold(s), []).append(i)
    ordered = sorted(groups.values(), key=lambda g: (len(g), g[0]), reverse=True)

    n_train = round((1 - test_frac) * len(smiles))
    train: list[int] = []
    test: list[int] = []
    for group in ordered:
        (train if len(train) + len(group) <= n_train else test).extend(group)
    return Split(np.array(sorted(train)), np.array(sorted(test)))


def random_split(labels, test_frac: float = 0.2, seed: int = 0) -> Split:
    """Stratified random split."""
    idx = np.arange(len(labels))
    train, test = train_test_split(idx, test_size=test_frac, stratify=labels, random_state=seed)
    return Split(np.sort(train), np.sort(test))
