import os

import pytest

from quant_research_agents.experiment import run_experiment
from quant_research_agents.models import ExperimentConfig
from quant_research_agents.store import ExperimentStore


@pytest.mark.skipif(
    not os.getenv("TEST_POSTGRES_URL"), reason="PostgreSQL integration requires TEST_POSTGRES_URL"
)
def test_postgres_round_trip_and_idempotency(tmp_path):
    store = ExperimentStore(tmp_path, os.environ["TEST_POSTGRES_URL"])
    result = run_experiment(ExperimentConfig(cost_bps=17))
    store.save(result)
    store.save(result)
    assert store.get(result["experiment_id"]) == result
    assert any(r["experiment_id"] == result["experiment_id"] for r in store.recent(20))
    assert store.get("' OR 1=1 --") is None
