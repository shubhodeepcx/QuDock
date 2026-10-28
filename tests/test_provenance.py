import json

import pandas as pd

from qudock import provenance
from qudock.cli import main


def test_record_has_code_data_and_environment(tmp_path):
    data = tmp_path / "bace.csv"
    data.write_text("mol,Class\nCCO,1\n")
    rec = provenance.record(command=["qudock", "data"], config={"seed": 0},
                            data_files=[data, tmp_path / "missing.csv"])
    assert rec["command"] == ["qudock", "data"] and rec["config"] == {"seed": 0}
    assert set(rec["data_files"]) == {"bace.csv"} and len(rec["data_files"]["bace.csv"]) == 64
    assert rec["packages"]["numpy"] and rec["python"]
    assert set(rec["git"]) == {"commit", "dirty"}


def test_dataset_hash_changes_with_content():
    a = pd.DataFrame({"smiles": ["CCO", "CCN"], "label": [0, 1]})
    b = a.assign(label=[1, 1])
    assert provenance.dataset_hash(a) == provenance.dataset_hash(a.copy())
    assert provenance.dataset_hash(a) != provenance.dataset_hash(b)


def test_cli_writes_provenance_next_to_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results" / "final").mkdir(parents=True)
    pd.DataFrame({"model": ["qsvc_pls", "rf_pls"] * 2, "n_train": [100] * 4, "seed": [0, 0, 1, 1],
                  "roc_auc": [0.70, 0.72, 0.71, 0.73]}).to_csv(
        tmp_path / "results" / "final" / "final_runs.csv", index=False)
    main(["scorecard", "--out", "card"])
    rec = json.loads((tmp_path / "card" / "provenance.json").read_text())
    assert rec["command"] == ["qudock", "scorecard", "--out", "card"]
    assert rec["config"]["out"] == "card"
