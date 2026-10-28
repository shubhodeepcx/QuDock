import numpy as np
import pandas as pd

from qudock.graphmap import (
    FAMILIES,
    PAIRS,
    GraphKernel,
    graph_angles,
    pharmacophore_counts,
    run_graph_study,
)


def test_counts_follow_the_chemistry():
    # benzoic acid: an aromatic ring and an acid; hexane: hydrophobic only
    raw = pharmacophore_counts(["OC(=O)c1ccccc1", "CCCCCC"])
    fam = dict(zip(FAMILIES, raw[0, :len(FAMILIES)]))
    assert fam["Aromatic"] == 1 and fam["NegIonizable"] == 1 and fam["Acceptor"] >= 1
    assert raw[1, FAMILIES.index("Aromatic")] == 0
    assert raw.shape == (2, len(FAMILIES) + len(PAIRS))


def test_closer_groups_couple_more_strongly():
    near, far = graph_angles(["NCc1ccccc1", "NCCCCCCc1ccccc1"])
    pair = PAIRS.index((FAMILIES.index("Donor"), FAMILIES.index("Aromatic")))
    assert near[len(FAMILIES) + pair] > far[len(FAMILIES) + pair]


def test_angles_are_bounded():
    A = graph_angles(["CC(=O)Nc1ccc(O)cc1", "NC(=N)c1ccccc1", "C"])
    assert A.min() >= 0 and A.max() < np.pi


def test_kernel_is_a_valid_gram_matrix():
    A = graph_angles(["CC(=O)Nc1ccc(O)cc1", "NC(=N)c1ccccc1", "OC(=O)c1ccccc1", "CCCC"])
    K = GraphKernel(reps=2, bandwidth=0.5)(A)
    assert np.allclose(np.diag(K), 1) and np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-9
    assert np.allclose(GraphKernel(bandwidth=0.0)(A), 1)       # no encoding, one state


def test_same_groups_in_a_different_arrangement_give_different_states():
    a, b = graph_angles(["NCc1ccccc1", "NCCCCCCc1ccccc1"])
    assert GraphKernel()(np.stack([a, b]))[0, 1] < 0.999


def test_graph_study_runs_on_toy_data():
    subs = ["O", "N", "C", "Cl", "F", "Br", "CO", "CN", "CC", "OC"]
    toy = pd.DataFrame({"smiles": ["c1ccccc1" + s for s in subs] + ["C1CCNCC1" + s for s in subs],
                        "label": [1, 0] * 10})
    out = run_graph_study(seeds=[0], sizes=[None], df=toy)
    assert set(out["model"]) == {"graph_quantum", "graph_rbf", "graph_rf", "qsvc_pls"}
    assert out["roc_auc"].between(0, 1).all()
