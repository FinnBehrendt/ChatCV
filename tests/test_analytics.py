import pytest
from fastapi.testclient import TestClient

from app.api import routes
from main import app


@pytest.mark.parametrize("analytics_error", [OSError, RuntimeError])
def test_chat_returns_answer_when_analytics_fails(mocker, monkeypatch, analytics_error):
    query = "What experience does the candidate have?"
    expected = {"answer": {"answer": "An example answer."}, "sources": []}
    completion = mocker.patch.object(
        routes, "get_chat_completion", return_value=expected
    )
    record_analytics = mocker.patch.object(
        routes.advanced_analytics,
        "log_chat_interaction_advanced",
        side_effect=analytics_error("Simulated analytics failure"),
    )
    # Keep the test independent of configured invite codes and real sessions.
    monkeypatch.setitem(
        app.dependency_overrides, routes.require_auth, lambda: "anonymous"
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/chat", json={"query": query})

    assert response.status_code == 200
    assert response.json() == expected
    completion.assert_called_once()
    record_analytics.assert_called_once()
