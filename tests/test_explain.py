import numpy as np
import pytest
from rdkit import Chem
from sklearn.svm import SVC

from qudock.explain import (
    bit_atoms,
    draw_explanation,
    environment_atoms,
    explain_molecule,
    substructure_smiles,
)
from qudock.features import FeaturePipeline, morgan_fingerprints
from qudock.kernels import KernelSVM, stack

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
# half the molecules carry a nitro group and are labelled active, so one
# substructure explains the model and the explanation has a right answer
ACTIVE = ["c1ccccc1[N+](=O)[O-]", "Cc1ccccc1[N+](=O)[O-]", "Clc1ccccc1[N+](=O)[O-]",
          "COc1ccccc1[N+](=O)[O-]", "CCc1ccccc1[N+](=O)[O-]", "Fc1ccccc1[N+](=O)[O-]"]
INACTIVE = ["c1ccccc1C", "Cc1ccccc1CC", "Clc1ccccc1C", "COc1ccccc1C", "CCc1ccccc1C",
            "Fc1ccccc1C"]


def _toy_model(fingerprints=False):
    smiles = ACTIVE + INACTIVE
    y = np.array([1] * len(ACTIVE) + [0] * len(INACTIVE))
    pipe = FeaturePipeline(n_qubits=3, method="pls", use_descriptors=True).fit(smiles, y)
    X = pipe.transform(smiles)
    if fingerprints:
        X = stack(X, morgan_fingerprints(smiles))
        model = KernelSVM(kind="tanimoto", n_angles=3, C=1).fit(X, y)
    else:
        model = SVC(kernel="rbf", C=10).fit(X, y)
    return pipe, model


def test_environment_atoms_grows_with_radius():
    mol = Chem.MolFromSmiles(ASPIRIN)
    assert environment_atoms(mol, 0, 0) == {0}
    assert environment_atoms(mol, 0, 0) < environment_atoms(mol, 0, 1)


def test_every_bit_names_at_least_one_atom_and_together_they_cover_the_molecule():
    mol = Chem.MolFromSmiles(ASPIRIN)
    atoms_of = bit_atoms(mol)
    assert atoms_of and all(atoms_of.values())
    assert set().union(*atoms_of.values()) == set(range(mol.GetNumAtoms()))


def test_substructure_smiles_names_a_single_atom_by_its_element():
    mol = Chem.MolFromSmiles(ASPIRIN)
    assert substructure_smiles(mol, {0}) == "C"
    assert len(substructure_smiles(mol, {4, 5, 6})) > 1


def test_one_weight_per_atom_and_the_score_is_the_model_score():
    pipe, model = _toy_model()
    smi = ACTIVE[0]
    exp = explain_molecule(smi, pipe, model)
    assert len(exp.atom_weights) == Chem.MolFromSmiles(smi).GetNumAtoms()
    assert exp.score == pytest.approx(model.decision_function(pipe.transform([smi]))[0])


def test_the_nitro_group_is_what_makes_the_toy_model_say_active():
    pipe, model = _toy_model()
    smi = ACTIVE[0]
    exp = explain_molecule(smi, pipe, model)
    mol = Chem.MolFromSmiles(smi)
    nitro = set(mol.GetSubstructMatch(Chem.MolFromSmarts("[N+](=O)[O-]")))
    best = int(np.argmax(exp.atom_weights))
    assert exp.score > 0
    assert best in nitro
    assert exp.atom_weights[sorted(nitro)].mean() > exp.atom_weights.mean()


def test_the_top_substructure_table_reads_the_right_way_round():
    pipe, model = _toy_model()
    exp = explain_molecule(ACTIVE[1], pipe, model)
    row = exp.top_substructures(1)[0]
    table = exp.table(3)
    assert len(table) == 3
    assert table.iloc[0]["Score change if removed"] == round(-row["attribution"], 3)


def test_a_stacked_model_is_explained_too():
    pipe, model = _toy_model(fingerprints=True)
    exp = explain_molecule(ACTIVE[0], pipe, model, fingerprints=True)
    assert np.abs(exp.atom_weights).sum() > 0


def test_drawing_returns_a_png():
    pipe, model = _toy_model()
    png = draw_explanation(explain_molecule(ACTIVE[0], pipe, model), size=(200, 150))
    assert png[:4] == b"\x89PNG"


def test_explain_study_runs_end_to_end_on_toy_data():
    import pandas as pd

    from qudock.explain import run_explain_study, summarise_explain

    toy = pd.DataFrame({"smiles": ACTIVE + INACTIVE + ["c1ccncc1C", "c1ccncc1[N+](=O)[O-]",
                                                       "C1CCNCC1C", "C1CCNCC1[N+](=O)[O-]"],
                        "label": [1] * 6 + [0] * 6 + [0, 1, 0, 1]})
    out = run_explain_study(n_qubits=2, n_train=12, n_molecules=2, df=toy)
    assert len(out) == 2
    assert (out["n_removed"] >= 1).all()
    assert set(summarise_explain(out)) >= {"mean_gap", "share_explanation_wins"}
