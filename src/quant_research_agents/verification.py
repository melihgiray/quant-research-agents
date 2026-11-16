"""Offline report checking against the current corpus and a fresh synthetic replay."""

import json
import math
from numbers import Real
from pathlib import Path
from typing import Literal

from .evidence import EvidenceDocument, content_digest
from .experiment import run_experiment
from .models import ExperimentConfig, Review, StrictModel
from .reporting import validate_review
from .retrieval import EvidenceIndex


class ReportPayload(StrictModel):
    mode: Literal["scripted", "live"]
    review: Review
    experiment: dict
    evidence: list[dict]
    macro_context: dict | None = None
    validation: str


def _same(actual, expected) -> bool:
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and actual.keys() == expected.keys()
            and all(_same(actual[k], v) for k, v in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_same(a, b) for a, b in zip(actual, expected, strict=True))
        )
    if isinstance(expected, Real) and not isinstance(expected, bool):
        return (
            isinstance(actual, Real)
            and not isinstance(actual, bool)
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)
        )
    return type(actual) is type(expected) and actual == expected


def verify_payload(payload: dict) -> dict:
    report = ReportPayload.model_validate(payload)
    known = {d["id"]: d for d in EvidenceIndex().documents}
    seen = set()
    for item in report.evidence:
        document = EvidenceDocument.model_validate(
            {k: item[k] for k in EvidenceDocument.model_fields}
        ).model_dump()
        identifier = document["id"]
        if identifier in seen or identifier not in known:
            raise ValueError("evidence IDs must be unique and present in the current corpus")
        seen.add(identifier)
        digest = content_digest(document)
        if digest != item.get("content_sha256") or digest != known[identifier]["content_sha256"]:
            raise ValueError("evidence content fingerprint does not match the current corpus")
    validate_review(report.review, report.experiment, report.evidence)
    manifest = report.experiment["manifest"]
    if manifest.get("synthetic") is not True:
        raise ValueError("offline replay supports only the bundled synthetic fixture")
    replay = run_experiment(ExperimentConfig.model_validate(manifest["config"]))
    if not _same(report.experiment, replay):
        raise ValueError("report experiment does not match a fresh synthetic replay")
    return {
        "status": "verified",
        "mode": report.mode,
        "experiment_id": replay["experiment_id"],
        "checks": [
            "structured numerical claims",
            "current corpus content",
            "synthetic replay within numerical tolerance",
        ],
        "limits": "Does not authenticate the author, prove retrieval occurred, validate external macro context, or verify free-form prose and citation entailment.",
    }


def verify_report(path: Path) -> dict:
    if path.stat().st_size > 10_000_000:
        raise ValueError("report exceeds the 10 MB offline verification limit")
    try:
        return verify_payload(json.loads(path.read_text()))
    except (KeyError, TypeError) as error:
        raise ValueError("report is missing required structured fields") from error
