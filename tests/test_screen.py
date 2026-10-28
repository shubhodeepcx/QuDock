import numpy as np
import pandas as pd
import pytest

from qudock.screen import (
    ScreenConfig,
    Screener,
    combine,
    drug_likeness,
    holdout_check,
    read_library,
    screen_library,
)

SUBS = ["O", "N", "C", "Cl", "F", "Br", "CO", "CN", "CC", "OC"]
TOY = pd.DataFrame({
    "smiles": ["c1ccccc1" + s for s in SUBS] + ["C1CCNCC1" + s for s in SUBS]
    + ["c1ccncc1" + s for s in SUBS],
    "label": [1, 0] * 15,
})
CFG = ScreenConfig(n_qubits=2)


@pytest.fixture(scope="module")
def screener():
    return Screener(CFG).fit(TOY, TOY)


def test_read_library_handles_csv_and_smi(tmp_path):
    csv = tmp_path / "lib.csv"
    pd.DataFrame({"SMILES": ["CCO", "nonsense"], "Name": ["ethanol", "bad"]}).to_csv(csv)
    lib = read_library(csv)
    assert list(lib["name"]) == ["ethanol", "bad"] and lib["smiles"].isna().tolist() == [False,
                                                                                        True]
    smi = tmp_path / "lib.smi"
    smi.write_text("c1ccccc1O phenol\nCCN\n")
    assert list(read_library(smi)["smiles"]) == ["Oc1ccccc1", "CCN"]
    (tmp_path / "x.csv").write_text("a\n1\n")
    with pytest.raises(ValueError, match="smiles"):
        read_library(tmp_path / "x.csv")


def test_drug_likeness_flags_pains():
    table = drug_likeness(["CC(=O)Nc1ccc(O)cc1", "O=C1C=CC(=O)C=C1"])   # paracetamol, quinone
    assert table["qed"].between(0, 1).all()
    assert table.loc[0, "pains"] == "" and table.loc[1, "pains"] != ""


def test_combined_score_is_a_weighted_geometric_mean():
    t = pd.DataFrame({"p_active": [1.0, 0.25], "bbb_probability": [1.0, 1.0], "qed": [1.0, 1.0]})
    got = combine(t, {"activity": 2, "bbb": 1, "qed": 1})
    assert got[0] == pytest.approx(1.0) and got[1] == pytest.approx(0.25 ** 0.5)


def test_screen_ranks_and_keeps_unparsable_rows(screener):
    lib = pd.DataFrame({"input": ["c1ccccc1O", "oops", "C1CCNCC1N"],
                        "smiles": ["Oc1ccccc1", None, "NC1CCNCC1"]})
    ranked = screen_library(screener, lib)
    assert list(ranked["rank"].iloc[:2]) == [1, 2] and ranked["combined"].iloc[0] >= \
        ranked["combined"].iloc[1]
    assert ranked["error"].iloc[-1] == "could not parse SMILES"
    good = ranked.iloc[:2]
    for col in ("p_active", "bbb_probability", "qed", "nn_similarity"):
        assert good[col].between(0, 1).all(), col
    assert set(good["activity_call"]) <= {"active", "inactive", "don't know",
                                          "unlike anything seen"}


def test_training_molecules_are_in_domain(screener):
    assert screener.score(TOY["smiles"][:5])["in_domain"].all()


def test_holdout_check_reports_both_rankings():
    table, summary = holdout_check(TOY, TOY, CFG)
    assert list(summary.index) == ["activity only", "combined rank"]
    assert "label" in table and np.isfinite(summary["roc_auc"]).all()
