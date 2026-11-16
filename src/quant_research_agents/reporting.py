"""Verify structured claims before creating a human-readable report."""

import html
import json
import math
from pathlib import Path

import pandas as pd

from .experiment import compute_metrics
from .models import Review


class ReviewValidationError(ValueError):
    """Model output can be corrected without changing the executed experiment."""


def validate_review(review: Review, result: dict, evidence: list[dict]) -> None:
    if review.status != "completed":
        raise ReviewValidationError("cannot publish a completed experiment with a refusal review")
    known = {item["id"] for item in evidence}
    if not review.citations or not set(review.citations) <= known:
        raise ReviewValidationError("review must cite retrieved evidence IDs only")
    if not review.claims:
        raise ReviewValidationError("review must include verifiable numerical claims")
    if {claim.variant for claim in review.claims} != {"baseline", "volatility_scaled"}:
        raise ReviewValidationError("review must include a numerical claim for both variants")
    recomputed = {
        name: compute_metrics(pd.Series([row["net"] for row in series]))
        for name, series in result["series"].items()
    }
    for variant, metrics in result["metrics"].items():
        for metric, value in metrics.items():
            actual = recomputed[variant][metric]
            if value is None and actual is None:
                continue
            if value is None or actual is None or not math.isclose(value, actual, abs_tol=1e-10):
                raise ValueError("stored metrics do not match return series")
    for claim in review.claims:
        actual = recomputed[claim.variant][claim.metric]
        if actual is None or not math.isclose(claim.value, actual, rel_tol=1e-5, abs_tol=5e-5):
            raise ReviewValidationError(
                f"unsupported numerical claim: {claim.variant}.{claim.metric}"
            )


def write_report(
    directory: Path,
    review: Review,
    result: dict,
    evidence: list[dict],
    mode: str,
    macro_context: dict | None = None,
) -> dict:
    validate_review(review, result, evidence)
    directory.mkdir(parents=True, exist_ok=False)
    manifest = result["manifest"]
    label = "SYNTHETIC DATA — engineering demonstration, not investment evidence"
    lines = [
        "# Quant Research Copilot",
        "",
        f"**{label}**",
        "",
        f"Execution mode: {mode}",
        "",
        f"Experiment: `{result['experiment_id']}`",
        "",
        f"Period: {manifest['evaluation_start']} to {manifest['evaluation_end']} "
        f"({manifest['observations']} observations)",
        "",
        "## Computed results",
        "",
        "| Variant | Annual return | Annual volatility | Sharpe | Max drawdown |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in result["metrics"].items():
        sharpe = f"{metrics['sharpe']:.4f}" if metrics["sharpe"] is not None else "undefined"
        lines.append(
            f"| {name} | {metrics['annual_return']:.2%} | "
            f"{metrics['annual_volatility']:.2%} | {sharpe} | {metrics['max_drawdown']:.2%} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        review.summary,
        "",
        "Automated checks verify structured numerical claims and citation IDs. "
        "They do not establish that sources entail every sentence of the interpretation.",
        "",
        "## Limitations",
        "",
    ]
    lines += [f"- {item}" for item in dict.fromkeys(result["limitations"] + review.limitations)]
    lines += ["", "## Retrieved evidence", ""]
    selected = [item for item in evidence if item["id"] in review.citations]
    lines += [f"- [{item['id']}] [{item['title']}]({item['source_url']})" for item in selected]
    if macro_context is not None:
        lines += [
            "",
            "## Optional current macro context",
            "",
            "This API context is separate from historical strategy inputs.",
            "",
            "```json",
            json.dumps(macro_context, indent=2),
            "```",
        ]
    lines += [
        "",
        "## Reproduction manifest",
        "",
        "```json",
        json.dumps(manifest, indent=2),
        "```",
        "",
    ]
    markdown = "\n".join(lines)
    (directory / "report.md").write_text(markdown)
    payload = {
        "mode": mode,
        "review": review.model_dump(),
        "experiment": result,
        "evidence": selected,
        "macro_context": macro_context,
        "validation": "numeric claims and retrieved citation IDs checked",
    }
    (directory / "report.json").write_text(json.dumps(payload, indent=2, allow_nan=False))
    # Escaped plain text keeps all model output inert. A native SVG plots actual
    # computed net-return streams, so viewing the report needs no network.
    curves = []
    for name, series in result["series"].items():
        equity = [1.0]
        for row in series:
            equity.append(equity[-1] * (1 + row["net"]))
        curves.append((name, equity))
    low = min(min(values) for _, values in curves)
    high = max(max(values) for _, values in curves)
    paths = []
    for (name, values), color in zip(curves, ["#54c6a5", "#eab66a"], strict=True):
        points = " ".join(
            f"{30 + i / (len(values) - 1) * 740:.2f},"
            f"{235 - (v - low) / max(high - low, 1e-9) * 200:.2f}"
            for i, v in enumerate(values)
        )
        paths.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    body = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quant Research Copilot · Experiment report</title>
<style>body{{background:#0c1720;color:#e6edf1;font:16px/1.65 system-ui;margin:0;padding:40px 24px}}
main{{max-width:1000px;margin:auto}}small{{color:#92a8b5}}h1{{font-size:36px;letter-spacing:-1px}}
.label{{color:#eab66a}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.7 ui-monospace,monospace}}
.chart{{background:#14232f;border-radius:12px;padding:20px;margin:28px 0}}svg{{width:100%}}
</style><main><small>QUANT RESEARCH AGENTS / EXPERIMENT RECORD</small>
<h1>Evidence before interpretation.</h1><p class="label">{label}</p>
<div class="chart"><strong>Growth of $1 after modeled costs</strong>
<p><span style="color:#54c6a5">Baseline</span> · <span style="color:#eab66a">Volatility scaled</span></p>
<svg viewBox="0 0 800 270" role="img" aria-label="Computed equity curves, baseline and volatility scaled">
<text x="30" y="20" fill="#92a8b5">${high:.3f}</text>{"".join(paths)}
<text x="30" y="262" fill="#92a8b5">{manifest["evaluation_start"]}</text>
<text x="650" y="262" fill="#92a8b5">{manifest["evaluation_end"]}</text></svg></div>
<pre>{html.escape(markdown)}</pre></main></html>"""
    (directory / "report.html").write_text(body)
    return {
        "markdown": str((directory / "report.md").resolve()),
        "html": str((directory / "report.html").resolve()),
        "json": str((directory / "report.json").resolve()),
    }
