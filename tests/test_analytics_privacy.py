import json
import logging

import pytest

from app.utils import analytics


@pytest.mark.parametrize("remote_tracking", [False, True])
def test_conversation_tracking_does_not_export_credentials(
    monkeypatch, mocker, tmp_path, caplog, remote_tracking
):
    log_path = tmp_path / "analytics.log"
    monkeypatch.setattr(analytics.config.data, "analytics_log_path", str(log_path))
    monkeypatch.setattr(analytics.config, "langsmith_api_key", None)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    recorder = analytics.AdvancedAnalytics()
    client = mocker.Mock() if remote_tracking else None
    recorder.langsmith_client = client
    token = "test-tracing-session-secret"
    code = "test-tracing-invite-secret"

    with caplog.at_level(logging.DEBUG, logger="chatcv.analytics"):
        recorder.start_chat_session(token, code, "Example Company")
        assert token in recorder.active_chat_runs
        recorder.end_chat_session(token)

    assert token not in recorder.active_chat_runs
    exported = caplog.text
    if client is not None:
        client.create_run.assert_called_once()
        client.update_run.assert_called_once()
        run_id = client.create_run.call_args.kwargs["id"]
        assert client.update_run.call_args.args[0] == run_id
        exported += repr(client.mock_calls)
    for secret in (token, code):
        assert secret not in exported


@pytest.mark.parametrize("success", [False, True])
def test_login_analytics_keep_outcome_but_not_credentials(
    monkeypatch, tmp_path, caplog, success
):
    log_path = tmp_path / "analytics.log"
    monkeypatch.setattr(analytics.config.data, "analytics_log_path", str(log_path))
    monkeypatch.setattr(analytics.config.logging, "analytics_enabled", True)
    secret = "test-login-analytics-secret"

    with caplog.at_level(logging.DEBUG, logger="chatcv.analytics"):
        analytics.log_login_event(secret, "Example Company", success)

    contents = log_path.read_text()
    entry = json.loads(contents.split(",", 1)[1])
    assert entry["type"] == "login"
    assert entry["success"] is success
    assert entry["company"] == "Example Company"
    assert secret not in contents + caplog.text


def test_job_matching_analytics_exclude_credentials_in_metadata(
    monkeypatch, tmp_path, caplog
):
    log_path = tmp_path / "analytics.log"
    monkeypatch.setattr(analytics.config.data, "analytics_log_path", str(log_path))
    monkeypatch.setattr(analytics.config, "langsmith_api_key", None)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    recorder = analytics.AdvancedAnalytics()
    token = "test-job-session-secret"
    code = "test-job-invite-secret"

    with caplog.at_level(logging.DEBUG, logger="chatcv.analytics"):
        recorder.log_job_matching_advanced(
            user_code=code,
            company="Example Company",
            job_source="text_input",
            analysis_text="Example analysis",
            response_time=0.1,
            session_token=token,
            metadata={
                "pipeline": "JobMatching",
                "llm_model": "example-model",
                "user_code": code,
                "session_token": token,
            },
        )

    contents = log_path.read_text()
    entry = json.loads(contents.split(",", 1)[1])
    assert entry["metadata"] == {
        "pipeline": "JobMatching",
        "llm_model": "example-model",
    }
    assert entry["company"] == "Example Company"
    for secret in (token, code):
        assert secret not in contents + caplog.text
