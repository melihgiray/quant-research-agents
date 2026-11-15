import asyncio
import copy
import json

import pytest

from quant_research_agents.models import Review
from quant_research_agents.reporting import validate_review
from quant_research_agents.runtime import DEFAULT_REQUEST, run_research


@pytest.mark.parametrize("architecture", ["single", "multi"])
def test_real_adk_tool_execution_report_and_trace(tmp_path, monkeypatch, architecture):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    result = asyncio.run(run_research(DEFAULT_REQUEST, tmp_path, architecture, "scripted"))
    assert result["status"] == "completed"
    assert result["mode"] == "scripted"
    assert result["total_tokens"] is None
    assert result["estimated_cost_usd"] is None
    assert result["tool_calls"] == ["search_methodology", "list_experiments", "run_backtest"]
    report = json.loads((tmp_path / "runs" / result["run_id"] / "report/report.json").read_text())
    assert report["experiment"]["manifest"]["synthetic"]
    assert len(report["experiment"]["metrics"]) == 2
    assert (tmp_path / "runs" / result["run_id"] / "trace.jsonl").exists()
    review = Review.model_validate(report["review"])
    bad = review.model_copy(deep=True)
    bad.claims[0].value += 1
    with pytest.raises(ValueError, match="unsupported numerical claim"):
        validate_review(bad, report["experiment"], report["evidence"])
    bad = review.model_copy(update={"citations": ["invented-source"]})
    with pytest.raises(ValueError, match="retrieved evidence"):
        validate_review(bad, report["experiment"], report["evidence"])
    corrupt = copy.deepcopy(report["experiment"])
    corrupt["series"]["baseline"][0]["net"] += 0.1
    with pytest.raises(ValueError, match="stored metrics"):
        validate_review(review, corrupt, report["evidence"])


def test_scripted_mode_cannot_pretend_to_answer_arbitrary_questions(tmp_path):
    with pytest.raises(ValueError, match="fixed demonstration"):
        asyncio.run(run_research("Buy stocks", tmp_path, mode="scripted"))


def test_live_mode_requires_credentials(tmp_path, monkeypatch):
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_GENAI_USE_VERTEXAI"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        asyncio.run(run_research(DEFAULT_REQUEST, tmp_path))


def test_false_model_claim_fails_without_publishing_report(tmp_path, monkeypatch):
    from quant_research_agents.demo_model import ScriptedDemoModel

    monkeypatch.delenv("DATABASE_URL", raising=False)

    class IncorrectReviewer(ScriptedDemoModel):
        async def generate_content_async(self, llm_request, stream=False):
            async for response in super().generate_content_async(llm_request, stream):
                if self.role == "reviewer":
                    review = json.loads(response.content.parts[0].text)
                    review["claims"][0]["value"] += 9
                    response.content.parts[0].text = json.dumps(review)
                yield response

    with pytest.raises(ValueError, match="unsupported numerical claim"):
        asyncio.run(
            run_research(
                DEFAULT_REQUEST,
                tmp_path,
                mode="scripted",
                model_factory=lambda role: IncorrectReviewer(role=role),
            )
        )
    runs = list((tmp_path / "runs").glob("*/run.json"))
    assert len(runs) == 1
    assert json.loads(runs[0].read_text())["status"] == "failed"
    assert not list((tmp_path / "runs").glob("*/report/report.md"))


def test_refused_request_does_not_execute_or_save_experiment(tmp_path, monkeypatch):
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    from quant_research_agents.demo_model import ScriptedDemoModel
    from quant_research_agents.models import ExperimentConfig, Plan

    monkeypatch.delenv("DATABASE_URL", raising=False)

    class RefusalModel(ScriptedDemoModel):
        async def generate_content_async(self, llm_request, stream=False):
            if self.role == "coordinator":
                text = Plan(
                    decision="refuse",
                    reason="Real data is unavailable.",
                    research_query="unsupported",
                    config=ExperimentConfig(),
                ).model_dump_json()
            elif self.role == "reviewer":
                text = Review(
                    status="refused",
                    summary="Real data is unavailable.",
                    citations=[],
                    claims=[],
                    limitations=["Synthetic fixture only."],
                ).model_dump_json()
            else:
                text = "Refused; no tools needed."
            yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))

    result = asyncio.run(
        run_research(
            DEFAULT_REQUEST,
            tmp_path,
            mode="scripted",
            model_factory=lambda role: RefusalModel(role=role),
        )
    )
    assert result["status"] == "refused"
    assert result["tool_calls"] == []
    assert not (tmp_path / "experiments.sqlite3").exists()
