"""Local setup checks. Never contact providers or print credential values."""

import os
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def diagnose(workspace: Path) -> dict:
    dependencies = {}
    for name in ("google-adk", "numpy", "pandas", "pydantic", "httpx", "psycopg"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    candidate = Path(workspace).resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    writable = candidate.is_dir() and os.access(candidate, os.W_OK)
    database = os.getenv("DATABASE_URL", "")
    database_valid = not database or database.startswith(("postgresql://", "postgres://"))
    postgres = bool(database) and database_valid
    credentials = {
        "gemini_key_present": bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
        "vertex_requested": os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true",
        "fred_key_present": bool(os.getenv("FRED_API_KEY")),
    }
    return {
        "python": platform.python_version(),
        "dependencies": dependencies,
        "workspace_access_check": writable,
        "database_backend": "postgresql" if postgres else "sqlite" if database_valid else "invalid",
        "database_driver_available": not postgres or dependencies["psycopg"] is not None,
        "credentials": credentials,
        "credential_validation": "not performed; presence does not establish validity or access",
        "network_requests": 0,
        "next_steps": [
            step
            for needed, step in (
                (not writable, "Select a writable workspace directory."),
                (not database_valid, "Use a PostgreSQL DATABASE_URL or omit it for SQLite."),
                (
                    postgres and not dependencies["psycopg"],
                    'Install the optional dependency: pip install -e ".[postgres]"',
                ),
                (
                    not credentials["gemini_key_present"] and not credentials["vertex_requested"],
                    "Set GOOGLE_API_KEY locally before live Gemini runs; the scripted demo needs no key.",
                ),
            )
            if needed
        ],
    }
