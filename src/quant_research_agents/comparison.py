"""Compare stored experiments only when dataset and scoring windows agree."""

import math
import re

import pandas as pd

from .experiment import compute_metrics
from .store import ExperimentStore


def _checked_metrics(result: dict) -> dict:
    computed = {
        name: compute_metrics(pd.Series([row["net"] for row in rows]))
        for name, rows in result["series"].items()
    }
    for name, metrics in computed.items():
        for key, value in metrics.items():
            stored = result["metrics"][name][key]
            if stored is None and value is None:
                continue
            if (
                stored is None
                or value is None
                or not math.isclose(stored, value, rel_tol=1e-10, abs_tol=1e-12)
            ):
                raise ValueError("stored metrics do not match return series")
    return computed


def compare_experiments(store: ExperimentStore, left_id: str, right_id: str) -> dict:
    if not all(re.fullmatch(r"[a-f0-9]{20}", identifier) for identifier in (left_id, right_id)):
        raise ValueError("experiment IDs must contain exactly 20 lowercase hexadecimal characters")
    left, right = store.get(left_id), store.get(right_id)
    missing = [
        identifier for identifier, result in ((left_id, left), (right_id, right)) if result is None
    ]
    if missing:
        return {"status": "not_found", "missing_ids": list(dict.fromkeys(missing))}
    keys = (
        "engine_version",
        "dataset_id",
        "data_sha256",
        "synthetic",
        "evaluation_start",
        "evaluation_end",
        "observations",
    )
    mismatches = [key for key in keys if left["manifest"][key] != right["manifest"][key]]
    changes = {
        key: {
            "left": left["manifest"]["config"].get(key),
            "right": right["manifest"]["config"].get(key),
        }
        for key in sorted(left["manifest"]["config"].keys() | right["manifest"]["config"].keys())
        if left["manifest"]["config"].get(key) != right["manifest"]["config"].get(key)
    }
    result = {"left_id": left_id, "right_id": right_id, "config_changes": changes}
    if mismatches:
        return {**result, "status": "incomparable", "mismatched_manifest_fields": mismatches}
    a, b = _checked_metrics(left), _checked_metrics(right)
    return {
        **result,
        "status": "compared",
        "left_metrics": a,
        "right_metrics": b,
        "delta_right_minus_left": {
            name: {
                key: b[name][key] - value
                if value is not None and b[name][key] is not None
                else None
                for key, value in metrics.items()
            }
            for name, metrics in a.items()
        },
        "limits": "Descriptive metric differences from stored returns. Not statistical significance or causal attribution. Returns and volatility use decimal units.",
    }
