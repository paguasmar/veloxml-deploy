from typer.testing import CliRunner
from veloxml.cli import app
from veloxml.orchestrator.skypilot import SkyPilotOrchestrator

runner = CliRunner()

def test_cli_status_clean_output(monkeypatch):
    monkeypatch.setattr(SkyPilotOrchestrator, "list_services", lambda: [])
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "No active VeloxML services" in result.stdout

def test_cli_status_with_services(monkeypatch):
    mock_services = [{
        "name": "smollm-test",
        "version": "1",
        "uptime": "5m",
        "status": "READY",
        "replicas": "1/1",
        "endpoint": "http://1.2.3.4:8000",
    }]
    monkeypatch.setattr(SkyPilotOrchestrator, "list_services", lambda: mock_services)
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "smollm-test" in result.stdout
    assert "READY" in result.stdout

def test_cli_down_all_flag(monkeypatch):
    calls = []
    def mock_teardown(self, purge_all=False):
        calls.append(purge_all)
    monkeypatch.setattr(SkyPilotOrchestrator, "teardown", mock_teardown)
    
    result = runner.invoke(app, ["down", "--name", "test-svc", "--all"])
    assert result.exit_code == 0
    assert calls == [True]
