import pandas as pd
import pytest

from qudock.data import clean, murcko_scaffold, random_split, scaffold_split

SMILES = [
    "c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "c1ccccc1Cl",  # benzene scaffold
    "C1CCCCC1O", "C1CCCCC1N",                              # cyclohexane scaffold
    "c1ccncc1C", "c1ccncc1O",                              # pyridine scaffold
    "c1ccc2ccccc2c1", "CCO",
]


def test_clean_drops_bad_and_duplicate_smiles():
    df = pd.DataFrame({
        "mol": ["c1ccccc1O", "Oc1ccccc1", "not_a_smiles", "CCN", "NCC", "CCC"],
        "Class": [1, 1, 0, 0, 1, 0],
    })
    out = clean(df)
    # phenol duplicate collapses to one row, the bad SMILES goes,
    # ethylamine has conflicting labels so both copies go
    assert sorted(out["smiles"]) == sorted(["Oc1ccccc1", "CCC"])
    assert set(out.columns) == {"smiles", "label"}


def test_scaffold_split_is_disjoint_and_complete():
    split = scaffold_split(SMILES, test_frac=0.3)
    assert set(split.train).isdisjoint(split.test)
    assert sorted([*split.train, *split.test]) == list(range(len(SMILES)))


def test_scaffold_split_keeps_scaffolds_together():
    split = scaffold_split(SMILES, test_frac=0.3)
    train_scaffolds = {murcko_scaffold(SMILES[i]) for i in split.train}
    test_scaffolds = {murcko_scaffold(SMILES[i]) for i in split.test}
    assert train_scaffolds.isdisjoint(test_scaffolds)


def test_random_split_is_stratified_and_seeded():
    labels = [0] * 50 + [1] * 50
    a = random_split(labels, test_frac=0.2, seed=1)
    b = random_split(labels, test_frac=0.2, seed=1)
    assert (a.test == b.test).all()
    assert sum(labels[i] for i in a.test) == 10


@pytest.mark.parametrize("frac", [0.1, 0.2, 0.3])
def test_scaffold_split_size_is_close_to_requested(frac):
    split = scaffold_split(SMILES, test_frac=frac)
    assert len(split.test) >= round(frac * len(SMILES)) - 1
