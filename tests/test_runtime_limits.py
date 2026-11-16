import asyncio
import json

import pytest
from google.adk.agents.invocation_context import LlmCallsLimitExceededError

from quant_research_agents.demo_model import ScriptedDemoModel
from quant_research_agents.models import RunLimits
from quant_research_agents.runtime import DEFAULT_REQUEST, run_research


def metadata(workspace):
    return json.loads(next((workspace / "runs").glob("*/run.json")).read_text())


def test_call_budget_stops_workflow_before_experiment(tmp_path):
    with pytest.raises(LlmCallsLimitExceededError):
        asyncio.run(
            run_research(
                DEFAULT_REQUEST, tmp_path, mode="scripted", limits=RunLimits(max_llm_calls=1)
            )
        )
    result = metadata(tmp_path)
    assert result["status"] == "failed"
    assert "run_backtest" not in result["tool_calls"]
    assert result["runtime_settings"]["max_llm_calls"] == 1
    assert not list(tmp_path.glob("runs/*/report/report.json"))


def test_timeout_stops_waiting_model_and_records_failure(tmp_path):
    class WaitingModel(ScriptedDemoModel):
        async def generate_content_async(self, llm_request, stream=False):
            await asyncio.Event().wait()
            yield

    with pytest.raises(TimeoutError):
        asyncio.run(
            run_research(
                DEFAULT_REQUEST,
                tmp_path,
                mode="scripted",
                model_factory=lambda role: WaitingModel(role=role),
                limits=RunLimits(timeout_seconds=1),
            )
        )
    assert metadata(tmp_path)["error_type"] == "TimeoutError"


def test_disabling_review_repair_fails_first_bad_claim(tmp_path):
    seen = []

    class BadReviewer(ScriptedDemoModel):
        async def generate_content_async(self, llm_request, stream=False):
            seen.append(self.role)
            async for response in super().generate_content_async(llm_request, stream):
                if self.role == "reviewer":
                    review = json.loads(response.content.parts[0].text)
                    review["claims"][0]["value"] += 1
                    response.content.parts[0].text = json.dumps(review)
                yield response

    with pytest.raises(ValueError, match="unsupported numerical claim"):
        asyncio.run(
            run_research(
                DEFAULT_REQUEST,
                tmp_path,
                mode="scripted",
                model_factory=lambda role: BadReviewer(role=role),
                limits=RunLimits(review_repair=False),
            )
        )
    assert "review_repair" not in seen
    assert metadata(tmp_path)["review_repair_attempts"] == 0


@pytest.mark.parametrize(
    "settings",
    [
        {"max_llm_calls": 0},
        {"max_llm_calls": 65},
        {"timeout_seconds": 601},
        {"timeout_seconds": float("nan")},
        {"max_llm_calls": True},
    ],
)
def test_invalid_limits_cannot_disable_bounds(settings):
    with pytest.raises(ValueError):
        RunLimits(**settings)
