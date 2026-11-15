"""An optional external API, kept separate from strategy inputs."""

import os
import time

import httpx

ALLOWED_SERIES = {"VIXCLS", "DGS10", "FEDFUNDS"}


def fetch_context(series_id: str, client: httpx.Client | None = None) -> dict:
    if series_id not in ALLOWED_SERIES:
        return {"status": "unsupported_series", "allowed": sorted(ALLOWED_SERIES)}
    key = os.environ.get("FRED_API_KEY")
    if not key:
        return {"status": "not_configured", "message": "Set FRED_API_KEY locally to fetch context."}
    owns_client = client is None
    client = client or httpx.Client(timeout=10, follow_redirects=False)
    try:
        for attempt in range(2):
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
                if response.status_code in (429, 500, 502, 503, 504) and attempt == 0:
                    time.sleep(0.25)
                    continue
                response.raise_for_status()
                observations = response.json()["observations"]
                return {
                    "status": "ok",
                    "series_id": series_id,
                    "source_url": f"https://fred.stlouisfed.org/series/{series_id}",
                    "usage": "Current report context only; not historical trading inputs.",
                    "observations": [
                        {"date": o["date"], "value": o["value"]} for o in observations
                    ],
                }
            except (httpx.HTTPError, ValueError, KeyError, TypeError):
                if attempt == 0:
                    time.sleep(0.25)
        # Never echo HTTP exceptions: their URL can contain the API key.
        return {"status": "unavailable", "message": "FRED failed after two bounded attempts."}
    finally:
        if owns_client:
            client.close()
