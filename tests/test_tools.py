import json
from types import SimpleNamespace

import httpx
import pytest

from quant_research_agents.fred import fetch_context
from quant_research_agents.models import ExperimentConfig
from quant_research_agents.retrieval import EvidenceIndex
from quant_research_agents.tools import ResearchTools


def test_retrieval_supports_abstention_and_stable_citations():
    index = EvidenceIndex()
    hits = index.search("momentum formation skip lookback")
    assert hits[0]["id"] == "momentum"
    assert hits[0]["source_url"].startswith("https://github.com/")
    assert index.search("zyxwvutsrqponmlkjihgfedcba") == []


def test_tools_require_evidence_and_history_and_write_idempotently(tmp_path):
    tools = ResearchTools(tmp_path)
    context = SimpleNamespace(state={})
    config = ExperimentConfig().model_dump()
    assert tools.run_backtest(**config, tool_context=context)["status"] == "invalid_experiment"
    tools.search_methodology("momentum volatility costs", context)
    assert tools.run_backtest(**config, tool_context=context)["status"] == "invalid_experiment"
    tools.list_experiments(5, context)
    result = tools.run_backtest(**config, tool_context=context)
    assert result["status"] == "ok"
    assert tools.run_backtest(**config, tool_context=context) == result
    assert len(tools.store.recent()) == 1
    assert (
        tools.run_backtest(**{**config, "cost_bps": 10}, tool_context=context)["status"]
        == "invalid_experiment"
    )


def test_coordinator_plan_is_enforced(tmp_path):
    tools = ResearchTools(tmp_path)
    context = SimpleNamespace(
        state={
            "plan": {
                "decision": "refuse",
                "reason": "Unsupported",
                "research_query": "momentum",
                "config": ExperimentConfig().model_dump(),
            }
        }
    )
    tools.search_methodology("momentum", context)
    tools.list_experiments(5, context)
    assert (
        tools.run_backtest(**ExperimentConfig().model_dump(), tool_context=context)["status"]
        == "invalid_experiment"
    )
    assert tools.store.recent() == []


def test_sql_queries_do_not_interpolate_user_input(tmp_path):
    store = ResearchTools(tmp_path).store
    assert store.get("' OR 1=1; DROP TABLE experiments; --") is None
    assert store.recent() == []
    with pytest.raises(ValueError):
        store.recent(100000)


def test_fred_missing_credentials_is_explicit(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    assert fetch_context("DGS10")["status"] == "not_configured"
    assert fetch_context("arbitrary-url")["status"] == "unsupported_series"


def test_fred_transient_failure_recovers_and_never_returns_key(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test-secret-do-not-echo")
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"observations": [{"date": "2026-11-13", "value": "4.1"}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = fetch_context("DGS10", client)
    assert len(calls) == 2
    assert result["status"] == "ok"
    assert "test-secret-do-not-echo" not in json.dumps(result)


def test_fred_permanent_failure_is_bounded_and_sanitized(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test-secret-do-not-echo")
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(403)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = fetch_context("DGS10", client)
    assert len(calls) == 1
    assert result["status"] == "unavailable"
    assert "test-secret-do-not-echo" not in json.dumps(result)


@pytest.mark.parametrize("value", ["NaN", "inf", "ignore instructions", 4.1, {}])
def test_fred_rejects_invalid_payload_without_retry(monkeypatch, value):
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"observations": [{"date": "2026-11-13", "value": value}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = fetch_context("DGS10", client)
    assert result["status"] == "invalid_response"
    assert len(calls) == 1


def test_fred_missing_observation_is_null(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"observations": [{"date": "2026-11-13", "value": "."}]}
            )
        )
    ) as client:
        result = fetch_context("DGS10", client)
    assert result["observations"] == [{"date": "2026-11-13", "value": None}]


def test_fred_transport_retry_is_bounded(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("private URL with key", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = fetch_context("DGS10", client)
    assert result == {"status": "unavailable", "attempts": 2, "reason": "transport_error"}
    assert len(calls) == 2
