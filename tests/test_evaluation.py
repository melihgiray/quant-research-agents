import asyncio
import json

import pytest

from quant_research_agents import evaluation
from quant_research_agents.evaluation import score_case


def test_refusal_after_execution_is_not_evaluation_success():
    score = score_case(
        {"expected_status": "refused"}, {"status": "refused", "tool_calls": ["run_backtest"]}
    )
    assert score["automated_pass"] is False


def test_completed_status_without_report_is_not_success():
    score = score_case(
        {"expected_status": "completed", "required_tools": ["run_backtest"]},
        {"status": "completed", "tool_calls": ["run_backtest"]},
    )
    assert score["automated_pass"] is False


def test_evaluation_isolates_trials_retains_failures_and_checkpoints(tmp_path, monkeypatch):
    source = tmp_path / "cases.json"
    source.write_text(
        json.dumps([{"id": "refuse", "prompt": "real prices", "expected_status": "refused"}])
    )
    workspaces = []

    async def fake_run(request, workspace, architecture, mode):
        assert mode == "live"
        assert request == "real prices"
        assert workspace not in workspaces
        assert not workspace.exists()
        workspaces.append(workspace)
        checkpoint = json.loads((workspace.parent / "evaluation.json").read_text())
        assert len(checkpoint["results"]) == len(workspaces) - 1
        if architecture == "single":
            run = workspace / "runs" / "fixture"
            run.mkdir(parents=True)
            (run / "run.json").write_text(
                json.dumps({"status": "failed", "run_id": "fixture", "total_tokens": 123})
            )
            raise RuntimeError("sensitive-provider-error")
        return {"status": "refused", "tool_calls": []}

    monkeypatch.setattr(evaluation, "run_research", fake_run)
    path = asyncio.run(evaluation.evaluate(source, tmp_path, trials=2, seed=7))
    first = json.loads(path.read_text())
    assert first["status"] == "completed"
    assert len(first["results"]) == first["scheduled_runs"] == 4
    failed = [r for r in first["results"] if not r["automated_pass"]]
    assert len(failed) == 2
    assert all(r["run"]["total_tokens"] == 123 for r in failed)
    assert "sensitive-provider-error" not in path.read_text()
    second = json.loads(
        asyncio.run(evaluation.evaluate(source, tmp_path, trials=2, seed=7)).read_text()
    )

    def order(result):
        return [(r["case_id"], r["trial"], r["architecture"]) for r in result["results"]]

    assert order(first) == order(second)
    assert first["cases_sha256"] == second["cases_sha256"]


def test_interrupted_evaluation_preserves_completed_attempts(tmp_path, monkeypatch):
    source = tmp_path / "cases.json"
    source.write_text(
        json.dumps([{"id": "refuse", "prompt": "real prices", "expected_status": "refused"}])
    )
    count = 0

    async def fake_run(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise asyncio.CancelledError()
        return {"status": "refused", "tool_calls": []}

    monkeypatch.setattr(evaluation, "run_research", fake_run)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(evaluation.evaluate(source, tmp_path))
    result = json.loads(next((tmp_path / "evaluations").glob("*/evaluation.json")).read_text())
    assert result["status"] == "interrupted"
    assert len(result["results"]) == 1


def test_evaluation_rejects_duplicate_cases_before_model_calls(tmp_path):
    source = tmp_path / "cases.json"
    case = {"id": "same", "prompt": "real prices", "expected_status": "refused"}
    source.write_text(json.dumps([case, case]))
    with pytest.raises(ValueError, match="unique"):
        asyncio.run(evaluation.evaluate(source, tmp_path))
    assert not (tmp_path / "evaluations").exists()
    with pytest.raises(ValueError, match="trials"):
        asyncio.run(evaluation.evaluate(source, tmp_path, trials=0))
