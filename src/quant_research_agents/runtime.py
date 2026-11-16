import asyncio
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from .agents import build_agent
from .demo_model import ScriptedDemoModel
from .models import RunLimits
from .provenance import capture_provenance

DEFAULT_REQUEST = (
    "Compare baseline momentum with volatility-scaled momentum using the bundled synthetic "
    "dataset, default parameters and 5 basis points of costs. Cite methodology and save a report."
)


async def run_research(
    request: str,
    workspace: Path,
    architecture="multi",
    mode="live",
    model_factory=None,
    database_url: str | None = None,
    limits: RunLimits | None = None,
) -> dict:
    """Execute one session. Omit database_url for SQLite in the given workspace."""
    limits = RunLimits.model_validate(limits or {})
    if mode not in {"live", "scripted"}:
        raise ValueError("mode must be live or scripted")
    if mode == "scripted" and request != DEFAULT_REQUEST:
        raise ValueError("scripted mode only supports the fixed demonstration request")
    if (
        mode == "live"
        and model_factory is None
        and not (
            os.getenv("GOOGLE_API_KEY")
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true"
        )
    ):
        raise ValueError("Set GOOGLE_API_KEY in your local .env before running Gemini.")
    run_id = uuid.uuid4().hex
    directory = Path(workspace) / "runs" / run_id
    directory.mkdir(parents=True)
    started = time.perf_counter()
    model_name = (
        "scripted-integration-fixture"
        if mode == "scripted"
        else os.getenv("QUANT_MODEL", "gemini-flash-latest")
    )
    metadata = {
        "schema_version": 1,
        "runtime_settings": limits.model_dump(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provenance": capture_provenance(),
        "run_id": run_id,
        "mode": mode,
        "architecture": architecture,
        "model": model_name,
        "request": request,
        "status": "running",
        "prompt_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "usage_observed": False,
        "estimated_cost_usd": None,
        "tool_calls": [],
        "review_repair_attempts": 0,
        "error_type": None,
    }
    sessions = InMemorySessionService()
    factory = (
        model_factory or (lambda role: ScriptedDemoModel(role=role))
        if mode == "scripted"
        else model_factory
    )
    runner = None
    try:
        agent = build_agent(
            Path(workspace),
            architecture,
            database_url=database_url,
            model_factory=factory,
            review_repair=limits.review_repair,
        )
        runner = Runner(app_name="quant_research_agents", node=agent, session_service=sessions)
        await sessions.create_session(
            app_name="quant_research_agents",
            user_id="local",
            session_id=run_id,
            state={"run_id": run_id, "mode": mode},
        )
        async with asyncio.timeout(limits.timeout_seconds):
            with (directory / "trace.jsonl").open("w") as trace:
                async for event in runner.run_async(
                    user_id="local",
                    session_id=run_id,
                    new_message=types.Content(role="user", parts=[types.Part(text=request)]),
                    run_config=RunConfig(max_llm_calls=limits.max_llm_calls),
                ):
                    calls = [
                        {"name": call.name, "args": call.args}
                        for call in event.get_function_calls()
                    ]
                    metadata["tool_calls"].extend(call["name"] for call in calls)
                    if "review_repair_attempts" in event.actions.state_delta:
                        metadata["review_repair_attempts"] = event.actions.state_delta[
                            "review_repair_attempts"
                        ]
                    usage = getattr(event, "usage_metadata", None)
                    if usage:
                        metadata["usage_observed"] = True
                        metadata["prompt_tokens"] += usage.prompt_token_count or 0
                        metadata["output_tokens"] += usage.candidates_token_count or 0
                        metadata["total_tokens"] += usage.total_token_count or 0
                    record = {
                        "event_id": event.id,
                        "timestamp": event.timestamp,
                        "author": event.author,
                        "text": "\n".join(
                            part.text
                            for part in (event.content.parts if event.content else [])
                            if part.text and not part.thought
                        )[:12000],
                        "calls": calls,
                        "responses": [
                            response.model_dump(mode="json")
                            for response in event.get_function_responses()
                        ],
                    }
                    trace.write(json.dumps(record, allow_nan=False) + "\n")
                    trace.flush()
        session = await sessions.get_session(
            app_name="quant_research_agents", user_id="local", session_id=run_id
        )
        outcome = session.state.get("outcome")
        if not outcome:
            raise RuntimeError("workflow ended without a validated report or explicit refusal")
        metadata.update(outcome)
        return metadata
    except asyncio.CancelledError:
        metadata.update(status="interrupted", error_type="CancelledError")
        raise
    except Exception as error:
        metadata.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        metadata["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        if not metadata["usage_observed"]:
            for key in ("prompt_tokens", "output_tokens", "total_tokens"):
                metadata[key] = None
        (directory / "run.json").write_text(json.dumps(metadata, indent=2, allow_nan=False))
        if runner is not None:
            await runner.close()
