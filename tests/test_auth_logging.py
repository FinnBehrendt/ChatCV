import logging

from fastapi import Request
from pytest import LogCaptureFixture, MonkeyPatch

from app.auth import auth


def test_generated_session_token_is_not_logged(caplog: LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="chatcv.auth"):
        token = auth.generate_session_token()

    assert isinstance(token, str)
    assert token
    assert token not in caplog.text


def test_valid_invite_code_is_not_logged(
    caplog: LogCaptureFixture, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(auth, "INVITE_CODES", {"valid_test_code": {"active": True}})
    monkeypatch.setattr(auth, "active_sessions", {})
    secret = "valid_test_code"
    with caplog.at_level(logging.DEBUG, logger="chatcv.auth"):
        token = auth.authenticate_with_code(code=secret)

    assert token
    assert isinstance(token, str)
    assert secret not in caplog.text
    assert token not in caplog.text
    assert auth.active_sessions[token] == secret


def test_invalid_invite_code_is_not_logged(
    caplog: LogCaptureFixture, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(auth, "INVITE_CODES", {})
    monkeypatch.setattr(auth, "active_sessions", {})
    secret = "invalid_test_code"
    with caplog.at_level(logging.DEBUG, logger="chatcv.auth"):
        token = auth.authenticate_with_code(code=secret)

    assert token is None
    assert secret not in caplog.text


def test_current_user_lookup_does_not_log_credentials(
    caplog: LogCaptureFixture, monkeypatch: MonkeyPatch
) -> None:
    session_token = "test-current-session-secret"
    invite_code = "test-current-invite-secret"
    other_session_token = "test-other-session-secret"
    other_invite_code = "test-other-invite-secret"
    monkeypatch.setattr(
        auth,
        "active_sessions",
        {
            session_token: invite_code,
            other_session_token: other_invite_code,
        },
    )
    # A real request with a synthetic cookie; no server or HTTP client is needed.
    request = Request(
        {
            "type": "http",
            "headers": [(b"cookie", f"session_token={session_token}".encode("ascii"))],
        }
    )

    with caplog.at_level(logging.DEBUG, logger="chatcv.auth"):
        user_code = auth.get_current_user(request)

    assert user_code == invite_code
    for secret in (
        session_token,
        invite_code,
        other_session_token,
        other_invite_code,
    ):
        assert secret not in caplog.text


def test_logout_removes_only_its_session_without_logging_credentials(
    caplog: LogCaptureFixture, monkeypatch: MonkeyPatch
) -> None:
    session_token = "test-logout-session-secret"
    invite_code = "test-logout-invite-secret"
    other_session_token = "test-unaffected-session-secret"
    other_invite_code = "test-unaffected-invite-secret"
    monkeypatch.setattr(
        auth,
        "active_sessions",
        {
            session_token: invite_code,
            other_session_token: other_invite_code,
        },
    )

    with caplog.at_level(logging.DEBUG, logger="chatcv.auth"):
        success = auth.logout_user(session_token)

    assert success is True
    assert auth.active_sessions == {other_session_token: other_invite_code}
    for secret in (
        session_token,
        invite_code,
        other_session_token,
        other_invite_code,
    ):
        assert secret not in caplog.text


def test_invalid_logout_preserves_sessions_without_logging_credentials(
    caplog: LogCaptureFixture, monkeypatch: MonkeyPatch
) -> None:
    invalid_token = "test-invalid-logout-secret"
    active_token = "test-existing-session-secret"
    invite_code = "test-existing-invite-secret"
    monkeypatch.setattr(auth, "active_sessions", {active_token: invite_code})

    with caplog.at_level(logging.DEBUG, logger="chatcv.auth"):
        success = auth.logout_user(invalid_token)

    assert success is False
    assert auth.active_sessions == {active_token: invite_code}
    for secret in (invalid_token, active_token, invite_code):
        assert secret not in caplog.text
