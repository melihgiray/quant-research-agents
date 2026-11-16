"""Portable source/environment fingerprints for reproducing agent runs."""

import hashlib
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .evidence import content_digest
from .retrieval import EvidenceIndex


def source_digest(root: Path) -> str:
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".json"} and "__pycache__" not in path.parts
    ]
    return content_digest(entries)


def capture_provenance() -> dict:
    package = Path(__file__).resolve().parent
    checkout = package.parent.parent
    git = {"commit": None, "dirty": None}
    if (checkout / ".git").exists():
        try:

            def command(*args):
                return subprocess.run(
                    ["git", "-C", str(checkout), *args],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=2,
                ).stdout.strip()

            git = {
                "commit": command("rev-parse", "HEAD"),
                "dirty": bool(command("status", "--porcelain")),
            }
        except (OSError, subprocess.SubprocessError):
            pass
    dependencies = {}
    for name in (
        "quant-research-agents",
        "google-adk",
        "google-genai",
        "numpy",
        "pandas",
        "pydantic",
    ):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    return {
        "python": platform.python_version(),
        "packages": dependencies,
        "git": git,
        "source_sha256": source_digest(package),
        "corpus_sha256": EvidenceIndex().corpus_sha256,
        "scope": "Package Python/JSON files and dependency versions; no credentials or environment dump.",
    }
