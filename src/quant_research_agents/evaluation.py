"""A small live evaluation harness; scripted demos are never quality benchmarks."""

import json
import uuid
from pathlib import Path

from .runtime import run_research


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


async def evaluate(cases_path: Path, workspace: Path) -> Path:
    cases = json.loads(cases_path.read_text())
    results = []
    for case in cases:
        for architecture in ("single", "multi"):
            try:
                outcome = await run_research(case["prompt"], workspace, architecture, mode="live")
                results.append(
                    {
                        "case_id": case["id"],
                        "architecture": architecture,
                        **score_case(case, outcome),
                        "run": outcome,
                    }
                )
            except Exception as error:
                results.append(
                    {
                        "case_id": case["id"],
                        "architecture": architecture,
                        "automated_pass": False,
                        "error_type": type(error).__name__,
                    }
                )
    destination = workspace / "evaluations" / f"{uuid.uuid4().hex}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "mode": "live",
                "cases": str(cases_path),
                "results": results,
                "limits": "Small development set, not a held-out benchmark. Automated checks do not "
                "measure citation entailment, prose accuracy, or planner reasoning. "
                "Review traces manually and repeat trials before comparing architectures.",
            },
            indent=2,
        )
    )
    return destination
