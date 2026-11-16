import json
import os
import subprocess
import sys
from pathlib import Path


def test_portfolio_walkthrough_runs_without_touching_configured_providers(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/demo_portfolio.py"
    result = subprocess.run(
        [sys.executable, str(script), "--workspace", str(tmp_path)],
        env={
            **os.environ,
            "GOOGLE_API_KEY": "unused-fixture-key",
            "DATABASE_URL": "postgresql://unused.invalid/db",
        },
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    output = json.loads(result.stdout)
    manifest = json.loads(Path(output["walkthrough"]).read_text())
    assert manifest["live_model_calls"] == 0
    for architecture in ("single", "multi"):
        assert manifest["architectures"][architecture]["verification"]["status"] == "verified"
        assert manifest["architectures"][architecture]["run"]["mode"] == "scripted"
    assert manifest["cost_comparison"]["config_changes"]["cost_bps"] == {"left": 5, "right": 20}
    assert Path(manifest["retrieval_file"]).exists()
