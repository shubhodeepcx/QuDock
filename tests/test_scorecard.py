import numpy as np
import pandas as pd
import pytest

from qudock.scorecard import build_scorecard, verdicts


def _runs(n_train, quantum_auc, classical_auc, seeds=4):
    rows = []
    for seed in range(seeds):
        wobble = 0.002 * (seed - seeds / 2)
        rows += [{"n_train": n_train, "seed": seed, "model": "hybrid_pls",
                  "roc_auc": quantum_auc + wobble},
                 {"n_train": n_train, "seed": seed, "model": "qsvc_pls",
                  "roc_auc": quantum_auc - 0.03 + wobble},
                 {"n_train": n_train, "seed": seed, "model": "rf_fingerprint",
                  "roc_auc": classical_auc + wobble},
                 {"n_train": n_train, "seed": seed, "model": "rf_pls",
                  "roc_auc": classical_auc - 0.04 + wobble}]
    return pd.DataFrame(rows)


def test_the_scorecard_picks_the_best_model_on_each_side():
    card = build_scorecard(_runs(400, 0.75, 0.78))
    assert len(card) == 1
    row = card.iloc[0]
    assert row["quantum_model"] == "hybrid_pls"
    assert row["classical_model"] == "rf_fingerprint"
    assert row["delta"] == pytest.approx(-0.03)
    assert row["setting"] == "BACE, 400 molecules"


def test_one_row_per_training_size():
    runs = pd.concat([_runs(100, 0.70, 0.74), _runs(400, 0.75, 0.78)])
    card = build_scorecard(runs)
    assert list(card["setting"]) == ["BACE, 100 molecules", "BACE, 400 molecules"]


def test_a_second_dataset_adds_its_full_size_row_only():
    bbbp = pd.concat([_runs(100, 0.70, 0.72), _runs(1572, 0.827, 0.829)])
    card = build_scorecard(_runs(400, 0.75, 0.78), bbbp_runs=bbbp)
    assert "BBBP, 1,572 molecules" in list(card["setting"])
    assert sum(s.startswith("BBBP") for s in card["setting"]) == 1


def test_a_setting_with_only_one_side_is_dropped():
    only_quantum = _runs(400, 0.75, 0.78)
    only_quantum = only_quantum[only_quantum["model"].isin(["hybrid_pls", "qsvc_pls"])]
    assert build_scorecard(only_quantum).empty


def test_the_trained_feature_map_is_compared_with_the_fixed_one():
    trainable = pd.DataFrame([{"n_train": n, "seed": s, "feature_map": m,
                               "roc_auc": 0.72 + (0.012 if m == "trained" else 0) + 0.001 * s}
                              for n in (100, 400) for s in range(5)
                              for m in ("fixed", "trained")])
    card = build_scorecard(_runs(400, 0.75, 0.78), trainable=trainable)
    row = card[card["setting"].str.startswith("Trained")].iloc[0]
    assert row["delta"] == pytest.approx(0.012)
    assert row["seeds"] == 10


def test_the_scaling_rows_carry_no_classical_arm():
    scaling = pd.DataFrame([{"n_qubits": q, "kind": k, "roc_auc": 0.70 + 0.01 * (k == "tuned")}
                            for q in (4, 6) for k in ("fixed", "tuned")])
    card = build_scorecard(_runs(400, 0.75, 0.78), scaling=scaling)
    rows = card[card["setting"].str.contains("qubits")]
    assert len(rows) == 2
    assert rows["delta"].isna().all()
    assert (rows["quantum_model"] == "tuned").all()


def test_a_clear_gap_is_a_win_and_a_small_one_is_a_tie():
    wide = build_scorecard(_runs(400, 0.60, 0.80))
    narrow = build_scorecard(_runs(400, 0.779, 0.780))
    assert verdicts(wide).iloc[0] == "classical wins"
    assert verdicts(narrow).iloc[0] == "tie"
    ahead = build_scorecard(_runs(400, 0.80, 0.60))
    assert verdicts(ahead).iloc[0] == "quantum wins"


def test_a_row_with_no_difference_measured_is_never_called_a_win():
    card = pd.DataFrame({"delta": [np.nan], "ci": [0.0]})
    assert verdicts(card).iloc[0] == "tie"
