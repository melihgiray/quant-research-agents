"""Live development evaluations with isolated databases and retained failures."""

import hashlib
import json
import random
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Literal

from pydantic import Field

from .models import RunLimits, StrictModel
from .runtime import run_research


class EvaluationCase(StrictModel):
    id: str = Field(min_length=1, max_length=100)
    prompt: str = Field(min_length=1, max_length=8000)
    expected_status: Literal["completed", "refused"]
    required_tools: list[str] = Field(default_factory=list)
    expected_config: dict = Field(default_factory=dict)


def score_case(case: dict, result: dict) -> dict:
    checks = {"expected_status": result["status"] == case["expected_status"]}
    if case["expected_status"] == "completed":
        checks["required_tools"] = set(case["required_tools"]) <= set(result["tool_calls"])
        if result.get("json"):
            report = json.loads(Path(result["json"]).read_text())
            checks["requested_configuration"] = all(
                report["experiment"]["manifest"]["config"].get(key) == value
                for key, value in case.get("expected_config", {}).items()
            )
        else:
            checks["requested_configuration"] = False
    else:
        checks["no_execution"] = "run_backtest" not in result["tool_calls"]
    return {"checks": checks, "automated_pass": all(checks.values())}


def _write_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)


async def evaluate(
    cases_path: Path,
    workspace: Path,
    trials: int = 1,
    seed: int = 42,
    limits: RunLimits | None = None,
) -> Path:
    limits = RunLimits.model_validate(limits or {})
    if not 1 <= trials <= 20:
        raise ValueError("trials must be between 1 and 20")
    source = cases_path.read_bytes()
    cases = [EvaluationCase.model_validate(case).model_dump() for case in json.loads(source)]
    if not cases or len(cases) > 100:
        raise ValueError("evaluation requires between 1 and 100 cases")
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("evaluation case IDs must be unique")
    schedule = [
        (case, architecture, trial)
        for case in cases
        for trial in range(1, trials + 1)
        for architecture in ("single", "multi")
    ]
    random.Random(seed).shuffle(schedule)
    directory = Path(workspace) / "evaluations" / uuid.uuid4().hex
    directory.mkdir(parents=True)
    destination = directory / "evaluation.json"
    payload = {
        "mode": "live",
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "cases_sha256": hashlib.sha256(source).hexdigest(),
        "cases": cases,
        "trials": trials,
        "runtime_settings": limits.model_dump(),
        "order_seed": seed,
        "scheduled_runs": len(schedule),
        "database_policy": "fresh SQLite database per case, trial and architecture",
        "versions": {name: version(name) for name in ("quant-research-agents", "google-adk")},
        "results": [],
        "limits": "Small development set, not a held-out benchmark. Automated checks do not "
        "measure citation entailment, prose accuracy, or planner reasoning. "
        "Review traces manually before comparing architectures.",
    }
    _write_checkpoint(destination, payload)
    try:
        for sequence, (case, architecture, trial) in enumerate(schedule):
            # Fresh databases prevent either architecture from seeing a previous
            # trial's experiments. DATABASE_URL intentionally does not apply here.
            run_workspace = directory / f"attempt-{sequence:04d}"
            entry = {
                "case_id": case["id"],
                "architecture": architecture,
                "trial": trial,
                "sequence": sequence,
            }
            try:
                outcome = await run_research(
                    case["prompt"], run_workspace, architecture, mode="live", limits=limits
                )
                entry.update(**score_case(case, outcome), run=outcome)
            except Exception as error:
                artifacts = list((run_workspace / "runs").glob("*/run.json"))
                entry.update(
                    automated_pass=False,
                    error_type=type(error).__name__,
                    run=json.loads(artifacts[0].read_text()) if artifacts else None,
                )
            payload["results"].append(entry)
            _write_checkpoint(destination, payload)
        payload["status"] = "completed"
    finally:
        if payload["status"] == "running":
            payload["status"] = "interrupted"
        payload["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_checkpoint(destination, payload)
    return destination
