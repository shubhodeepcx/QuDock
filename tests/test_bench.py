import pandas as pd
import pytest

from qudock.bench import BUILTIN, leaderboard, resolve, run_bench

SUBS = ["O", "N", "C", "Cl", "F", "Br", "CO", "CN", "CC", "OC"]
TOY = pd.DataFrame({
    "smiles": ["c1ccccc1" + s for s in SUBS] + ["C1CCNCC1" + s for s in SUBS]
    + ["c1ccncc1" + s for s in SUBS],
    "label": [1, 0] * 15,
})


@pytest.mark.parametrize("name", sorted(BUILTIN))
def test_every_builtin_model_runs(name):
    out = run_bench([name], tasks=["bace"], seeds=[0], datasets={"bace": TOY})
    assert len(out) == 1 and 0 <= out["roc_auc"].iloc[0] <= 1


def test_plugins_load_by_module_path_and_leaderboard_ranks(monkeypatch):
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    out = run_bench(["bench_plugin:nearest", "rf_fingerprint"], tasks=["bace", "bbbp"],
                    seeds=[0, 1], datasets={"bace": TOY, "bbbp": TOY})
    assert len(out) == 8
    board = leaderboard(out)
    assert set(board.index) == {"bench_plugin:nearest", "rf_fingerprint"}
    assert {"mean_rank", "quantum", "bace", "bbbp"} <= set(board.columns)


def test_unknown_model_is_explained():
    with pytest.raises(ValueError, match="module.path:function"):
        resolve("not_a_model")
