import pandas as pd

from qudock.cost import SECONDS_PER_CIRCUIT, circuits_needed, cost_table, device_seconds


def test_a_fidelity_kernel_needs_one_circuit_per_pair():
    # 4 training molecules: 10 pairs including the diagonal, plus 4 * 2 to score 2 test molecules
    assert circuits_needed("fidelity", 4, 2) == 18
    assert circuits_needed("fidelity", 1210, 303) == 1210 * 1211 // 2 + 1210 * 303


def test_a_projected_kernel_needs_one_circuit_per_molecule():
    assert circuits_needed("projected", 1210, 303) == 1513


def test_classical_models_need_no_circuits():
    assert circuits_needed("classical", 1210, 303) == 0
    assert device_seconds(0) == 0


def test_device_time_uses_the_rate_we_measured():
    # our own job: 66 circuits in 20 seconds of quantum time
    assert device_seconds(66) == 20
    assert SECONDS_PER_CIRCUIT < 1


def test_the_fidelity_kernel_is_the_expensive_one_at_full_size():
    fidelity = circuits_needed("fidelity", 1210, 303)
    assert device_seconds(fidelity) / 3600 > 90         # over ninety hours of device time
    assert fidelity > 600 * circuits_needed("projected", 1210, 303)


def test_cost_table_keeps_only_the_full_size_rows():
    results = pd.DataFrame({
        "model": ["a", "b", "a", "b"], "kind": ["fidelity", "classical"] * 2,
        "n_train": [100, 100, 400, 400], "roc_auc": [0.7, 0.8, 0.75, 0.85],
        "total_seconds": [10.0, 1.0, 40.0, 2.0], "circuits": [5000, 0, 80000, 0],
        "device_hours": [0.4, 0.0, 6.7, 0.0]})
    table = cost_table(results)
    assert len(table) == 2
    assert table.iloc[0]["model"] == "b"
    assert table["roc_auc"].tolist() == [0.85, 0.75]
