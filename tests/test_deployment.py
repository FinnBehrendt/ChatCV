"""Deployment constraints through the application's real startup lifecycle."""

import sys

import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from main import app


@pytest.fixture(autouse=True)
def isolated_launch_configuration(monkeypatch: MonkeyPatch) -> None:
    for variable in ("WEB_CONCURRENCY", "UVICORN_WORKERS", "GUNICORN_CMD_ARGS"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(sys, "argv", ["uvicorn", "main:app"])


@pytest.mark.parametrize("variable", ["WEB_CONCURRENCY", "UVICORN_WORKERS"])
def test_startup_rejects_multiple_workers(
    monkeypatch: MonkeyPatch, variable: str
) -> None:
    monkeypatch.setenv(variable, "2")
    with pytest.raises(RuntimeError, match="single worker"), TestClient(app) as client:
        client.get("/health")


@pytest.mark.parametrize("source", ["argv", "GUNICORN_CMD_ARGS"])
@pytest.mark.parametrize(
    "arguments", [["--workers", "2"], ["--workers=2"], ["-w", "2"], ["-w2"]]
)
def test_startup_rejects_multi_worker_launch_arguments(
    monkeypatch: MonkeyPatch, source: str, arguments: list[str]
) -> None:
    if source == "argv":
        monkeypatch.setattr(sys, "argv", ["uvicorn", "main:app", *arguments])
    else:
        monkeypatch.setenv(source, " ".join(arguments))
    with pytest.raises(RuntimeError, match="single worker"), TestClient(app) as client:
        client.get("/health")


@pytest.mark.parametrize("launch", ["default", "environment", "argv", "gunicorn"])
def test_supported_single_worker_startup_serves_health(
    monkeypatch: MonkeyPatch, launch: str
) -> None:
    if launch == "environment":
        monkeypatch.setenv("WEB_CONCURRENCY", "1")
        monkeypatch.setenv("UVICORN_WORKERS", "1")
    elif launch == "argv":
        monkeypatch.setattr(sys, "argv", ["uvicorn", "main:app", "--workers=1"])
    elif launch == "gunicorn":
        monkeypatch.setenv("GUNICORN_CMD_ARGS", "--bind '127.0.0.1:8000' -w1")
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200


@pytest.mark.parametrize("value", ["0", "-1", "two"])
def test_invalid_worker_count_fails_at_startup(
    monkeypatch: MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv("WEB_CONCURRENCY", value)
    with pytest.raises(RuntimeError, match="single worker"), TestClient(app) as client:
        client.get("/health")
