import asyncio
import json

import pandas as pd
import pytest

from quant_research_agents.experiment import compute_metrics
from quant_research_agents.runtime import DEFAULT_REQUEST, run_research
from quant_research_agents.verification import verify_payload, verify_report


@pytest.fixture
def report(tmp_path):
    result = asyncio.run(run_research(DEFAULT_REQUEST, tmp_path, mode="scripted"))
    return json.loads(open(result["json"]).read())


def test_offline_verification_replays_saved_experiment(report, tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    before = path.read_bytes()
    assert verify_report(path)["status"] == "verified"
    assert path.read_bytes() == before


@pytest.mark.parametrize("field", ["text", "source_url", "content_sha256"])
def test_modified_evidence_is_rejected(report, field):
    report["evidence"][0][field] = (
        "https://example.com/changed" if field == "source_url" else "changed"
    )
    with pytest.raises(ValueError, match="fingerprint"):
        verify_payload(report)


def test_coherently_fabricated_returns_metrics_and_claims_fail_replay(report):
    experiment = report["experiment"]
    experiment["series"]["baseline"][0]["net"] += 0.01
    experiment["metrics"]["baseline"] = compute_metrics(
        pd.Series([r["net"] for r in experiment["series"]["baseline"]])
    )
    for claim in report["review"]["claims"]:
        claim["value"] = experiment["metrics"][claim["variant"]][claim["metric"]]
    with pytest.raises(ValueError, match="fresh synthetic replay"):
        verify_payload(report)
