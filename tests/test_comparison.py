import pytest

from quant_research_agents.comparison import compare_experiments
from quant_research_agents.experiment import run_experiment
from quant_research_agents.models import ExperimentConfig
from quant_research_agents.store import ExperimentStore


def test_comparison_reports_cost_change_and_recomputed_delta(tmp_path):
    store = ExperimentStore(tmp_path)
    left = run_experiment(ExperimentConfig(cost_bps=0))
    right = run_experiment(ExperimentConfig(cost_bps=20))
    for result in (left, right):
        store.save(result)
    comparison = compare_experiments(store, left["experiment_id"], right["experiment_id"])
    assert comparison["status"] == "compared"
    assert comparison["config_changes"] == {"cost_bps": {"left": 0, "right": 20}}
    assert comparison["delta_right_minus_left"]["baseline"]["annual_return"] < 0
    assert len(store.recent()) == 2


def test_different_windows_are_not_compared(tmp_path):
    store = ExperimentStore(tmp_path)
    results = [run_experiment(ExperimentConfig(lookback=lookback)) for lookback in (252, 504)]
    for result in results:
        store.save(result)
    comparison = compare_experiments(store, *(r["experiment_id"] for r in results))
    assert comparison["status"] == "incomparable"
    assert "evaluation_start" in comparison["mismatched_manifest_fields"]
    assert "delta_right_minus_left" not in comparison


def test_missing_and_invalid_ids_do_not_return_other_records(tmp_path):
    store = ExperimentStore(tmp_path)
    assert compare_experiments(store, "a" * 20, "b" * 20)["status"] == "not_found"
    with pytest.raises(ValueError, match="hexadecimal"):
        compare_experiments(store, "' OR 1=1 --", "b" * 20)
