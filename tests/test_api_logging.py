import json
import logging
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import routes
from app.auth import auth
from app.utils import analytics

INVITE_CODE = "test-api-invite-secret"
SESSION_TOKEN = "test-api-session-secret"
INVALID_CODE = "test-api-invalid-secret"
COMPANY = "Example Company"


@pytest.fixture
def api_environment(monkeypatch, mocker, tmp_path, caplog):
    """Exercise real auth and analytics with isolated state and no remote client."""
    log_path = tmp_path / "analytics.log"
    monkeypatch.setattr(
        auth, "INVITE_CODES", {INVITE_CODE: {"active": True, "company": COMPANY}}
    )
    monkeypatch.setattr(auth, "active_sessions", {})
    monkeypatch.setattr(analytics.config, "invite_codes", {})
    monkeypatch.setattr(analytics.config.data, "analytics_log_path", str(log_path))
    monkeypatch.setattr(analytics.config.logging, "analytics_enabled", True)
    monkeypatch.setattr(analytics.config, "langsmith_api_key", None)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    recorder = analytics.AdvancedAnalytics()
    monkeypatch.setattr(routes, "advanced_analytics", recorder)
    mocker.patch.object(auth.secrets, "token_urlsafe", return_value=SESSION_TOKEN)
    for name in ("chatcv.auth", "chatcv.api", "chatcv.analytics"):
        caplog.set_level(logging.DEBUG, logger=name)

    app = FastAPI()
    app.include_router(routes.router)
    with TestClient(app, base_url="http://localhost") as client:
        yield SimpleNamespace(client=client, recorder=recorder, log_path=log_path)


def assert_no_credentials(environment, caplog):
    recorded = caplog.text
    if environment.log_path.exists():
        recorded += environment.log_path.read_text()
    for secret in (INVITE_CODE, SESSION_TOKEN, INVALID_CODE):
        assert secret not in recorded


def test_login_does_not_log_credentials(api_environment, caplog):
    response = api_environment.client.post(
        "/auth/login", json={"invite_code": INVITE_CODE}
    )

    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.cookies["session_token"] == SESSION_TOKEN
    assert auth.active_sessions[SESSION_TOKEN] == INVITE_CODE
    assert SESSION_TOKEN in api_environment.recorder.active_chat_runs
    assert_no_credentials(api_environment, caplog)


def test_failed_login_does_not_log_submitted_code(api_environment, caplog):
    response = api_environment.client.post(
        "/auth/login", json={"invite_code": INVALID_CODE}
    )

    assert response.status_code == 401
    assert not auth.active_sessions
    assert "session_token" not in response.cookies
    assert_no_credentials(api_environment, caplog)


def test_status_and_logout_do_not_log_credentials(api_environment, caplog):
    login = api_environment.client.post(
        "/auth/login", json={"invite_code": INVITE_CODE}
    )
    assert login.status_code == 200
    # Check these routes independently of the login logs and file entry.
    caplog.clear()
    api_environment.log_path.write_text("")

    status = api_environment.client.get("/auth/status")
    assert status.status_code == 200
    assert status.json()["authenticated"] is True
    assert_no_credentials(api_environment, caplog)

    logout = api_environment.client.post("/auth/logout")
    assert logout.status_code == 200
    assert logout.json()["success"] is True
    assert SESSION_TOKEN not in auth.active_sessions
    assert SESSION_TOKEN not in api_environment.recorder.active_chat_runs
    assert "session_token" not in api_environment.client.cookies
    assert_no_credentials(api_environment, caplog)


@pytest.mark.parametrize("endpoint", ["/chat", "/summary", "/job-match"])
def test_generation_routes_exclude_credentials_from_telemetry(
    api_environment, caplog, mocker, endpoint
):
    expected_results = {
        "/chat": {"answer": {"answer": "Example answer"}, "sources": []},
        "/summary": {"summary_md": "Example summary"},
        "/job-match": {"analysis": "Example analysis", "job_source": "text_input"},
    }
    function_names = {
        "/chat": "get_chat_completion",
        "/summary": "get_auto_summary",
        "/job-match": "analyze_job_match",
    }
    model_call = mocker.patch.object(
        routes, function_names[endpoint], return_value=expected_results[endpoint]
    )
    mocker.patch.object(routes, "process_job_description", return_value="Example job")
    login = api_environment.client.post(
        "/auth/login", json={"invite_code": INVITE_CODE}
    )
    assert login.status_code == 200
    caplog.clear()
    api_environment.log_path.write_text("")

    response = api_environment.client.post(
        endpoint, json={"query": "Example question", "text": "Example job"}
    )

    assert response.status_code == 200
    for key, value in expected_results[endpoint].items():
        if key != "job_source":
            assert response.json()[key] == value
    model_call.assert_called_once()
    # These arguments become inputs to the traceable model methods.
    trace_inputs = repr(model_call.call_args)
    for secret in (INVITE_CODE, SESSION_TOKEN):
        assert secret not in trace_inputs
    assert_no_credentials(api_environment, caplog)
    entries = [
        json.loads(line.split(",", 1)[1])
        for line in api_environment.log_path.read_text().splitlines()
    ]
    assert len(entries) == 1
    assert entries[0]["company"] == COMPANY
