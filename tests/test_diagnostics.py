import json

from quant_research_agents.diagnostics import diagnose


def test_diagnostics_never_echo_credentials_or_touch_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "private-gemini-key")
    monkeypatch.setenv("FRED_API_KEY", "private-fred-key")
    monkeypatch.setenv("DATABASE_URL", "postgresql://owner:private-password@localhost/db")
    workspace = tmp_path / "not-created"
    result = diagnose(workspace)
    assert result["credentials"]["gemini_key_present"]
    assert result["database_backend"] == "postgresql"
    assert result["network_requests"] == 0
    assert "private-" not in json.dumps(result)
    assert not workspace.exists()


def test_invalid_database_and_file_workspace_are_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "https://wrong:secret@host")
    target = tmp_path / "file"
    target.write_text("original")
    result = diagnose(target)
    assert not result["workspace_access_check"]
    assert result["database_backend"] == "invalid"
    assert target.read_text() == "original"
