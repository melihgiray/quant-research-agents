"""Optional FRED context with bounded transient retries and validated observations."""

import math
import os
import time
from datetime import date

import httpx

ALLOWED_SERIES = {"VIXCLS", "DGS10", "FEDFUNDS"}
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def _observations(payload: dict) -> list[dict]:
    rows = payload["observations"]
    if not isinstance(rows, list) or len(rows) > 10:
        raise ValueError("invalid observation count")
    result = []
    dates = set()
    for row in rows:
        day, value = row["date"], row["value"]
        if not isinstance(day, str) or date.fromisoformat(day).isoformat() != day or day in dates:
            raise ValueError("invalid observation date")
        dates.add(day)
        if value == ".":
            value = None
        elif not isinstance(value, str) or not math.isfinite(float(value)):
            raise ValueError("invalid observation value")
        result.append({"date": day, "value": value})
    return result


def fetch_context(series_id: str, client: httpx.Client | None = None) -> dict:
    if series_id not in ALLOWED_SERIES:
        return {"status": "unsupported_series", "allowed": sorted(ALLOWED_SERIES)}
    key = os.environ.get("FRED_API_KEY")
    if not key:
        return {"status": "not_configured", "message": "Set FRED_API_KEY locally to fetch context."}
    owns_client = client is None
    client = client or httpx.Client(timeout=10, follow_redirects=False)
    try:
        for attempt in range(1, 3):
            try:
                response = client.get(
                    "https://api.stlouisfed.org/fred/series/observations",
                    params={
                        "series_id": series_id,
                        "api_key": key,
                        "file_type": "json",
                        "sort_order": "desc",
                        "limit": 10,
                    },
                )
            except httpx.TransportError:
                if attempt == 1:
                    time.sleep(0.25)
                    continue
                return {"status": "unavailable", "attempts": attempt, "reason": "transport_error"}
            if response.status_code in RETRYABLE_STATUS and attempt == 1:
                time.sleep(0.25)
                continue
            if response.status_code != 200:
                return {
                    "status": "unavailable",
                    "attempts": attempt,
                    "reason": "http_error",
                    "http_status": response.status_code,
                }
            try:
                observations = _observations(response.json())
            except (ValueError, KeyError, TypeError, OverflowError):
                return {"status": "invalid_response", "attempts": attempt}
            return {
                "status": "ok",
                "attempts": attempt,
                "series_id": series_id,
                "source_url": f"https://fred.stlouisfed.org/series/{series_id}",
                "usage": "Current report context only; not historical trading inputs. Missing observations are null.",
                "observations": observations,
            }
    finally:
        if owns_client:
            client.close()
