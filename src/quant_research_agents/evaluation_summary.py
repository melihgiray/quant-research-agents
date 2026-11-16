"""Descriptive evaluation summaries that preserve failures and missing usage."""

import math
from collections import Counter
from statistics import median


def _nonnegative(value, name):
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite nonnegative number or null")
    return value


def summarize_evaluation(payload: dict) -> dict:
    if payload.get("mode") != "live":
        raise ValueError("agent quality summaries require a live evaluation artifact")
    results = payload["results"]
    groups = {}
    for architecture in sorted({r["architecture"] for r in results}):
        entries = [r for r in results if r["architecture"] == architecture]
        runs = [r["run"] for r in entries if r.get("run") is not None]
        if any(r.get("mode") != "live" for r in runs):
            raise ValueError(
                "cannot summarize scripted or unlabeled runs as live model evaluations"
            )
        durations = [
            value
            for r in runs
            if (value := _nonnegative(r.get("elapsed_seconds"), "elapsed_seconds")) is not None
        ]
        tokens = [
            value
            for r in runs
            if r.get("usage_observed")
            and (value := _nonnegative(r.get("total_tokens"), "total_tokens")) is not None
        ]
        costs = [
            value
            for r in runs
            if (value := _nonnegative(r.get("estimated_cost_usd"), "estimated_cost_usd"))
            is not None
        ]
        repairs = [
            value
            for r in runs
            if (value := _nonnegative(r.get("review_repair_attempts"), "review_repair_attempts"))
            is not None
        ]
        passed = sum(r.get("automated_pass") is True for r in entries)
        groups[architecture] = {
            "attempted": len(entries),
            "automated_passes": passed,
            "automated_pass_rate": passed / len(entries),
            "statuses": dict(
                Counter((r.get("run") or {}).get("status", "unrecorded") for r in entries)
            ),
            "models": sorted({r["model"] for r in runs if r.get("model")}),
            "latency_observations": len(durations),
            "median_seconds": median(durations) if durations else None,
            "p95_seconds": sorted(durations)[math.ceil(0.95 * len(durations)) - 1]
            if durations
            else None,
            "usage_observations": len(tokens),
            "usage_coverage": len(tokens) / len(entries),
            "observed_total_tokens": sum(tokens) if tokens else None,
            "cost_observations": len(costs),
            "observed_cost_usd": sum(costs) if costs else None,
            "repair_observations": len(repairs),
            "repair_attempt_rate": sum(r > 0 for r in repairs) / len(repairs) if repairs else None,
        }
    return {
        "mode": "live",
        "evaluation_status": payload.get("status", "unknown"),
        "recorded_attempts": len(results),
        "scheduled_runs": payload.get("scheduled_runs"),
        "architectures": groups,
        "limits": "Descriptive development-set statistics, not evidence of superiority. All recorded failures remain in pass-rate denominators. Latency uses recorded durations, including failures; p95 is nearest-rank. Token/cost sums cover observed values only, never missing usage. Check model, source and runtime settings before comparison.",
    }
