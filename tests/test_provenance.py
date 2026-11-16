import json

from quant_research_agents.provenance import capture_provenance, source_digest


def test_source_digest_changes_with_code_but_ignores_generated_cache(tmp_path):
    (tmp_path / "main.py").write_text("a = 1\n")
    first = source_digest(tmp_path)
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__/noise.py").write_text("generated")
    assert source_digest(tmp_path) == first
    (tmp_path / "main.py").write_text("a = 2\n")
    assert source_digest(tmp_path) != first


def test_provenance_has_no_environment_secrets(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "do-not-record-secret")
    result = capture_provenance()
    assert len(result["source_sha256"]) == len(result["corpus_sha256"]) == 64
    assert result["packages"]["google-adk"]
    assert "do-not-record-secret" not in json.dumps(result)
