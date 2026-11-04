"""Resolve the public Compose configuration without starting containers."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("custom_limits", [False, True])
def test_compose_forwards_conversation_limits(
    tmp_path: Path, custom_limits: bool
) -> None:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker Compose CLI is needed to resolve deployment configuration")
    environment = {
        key: os.environ[key]
        for key in ("PATH", "HOME", "DOCKER_CONFIG")
        if key in os.environ
    }
    environment["GUARDRAILS_TOKEN"] = "synthetic-token"
    version = subprocess.run(
        [docker, "compose", "version"],
        check=False,
        env=environment,
        capture_output=True,
        timeout=20,
    )
    if version.returncode:
        pytest.skip("Docker Compose plugin is unavailable")
    dotenv = tmp_path / "compose.env"
    dotenv.write_text(
        "CONVERSATION_MAX_TURNS=3\nCONVERSATION_IDLE_MINUTES=7\nCONVERSATION_CAPACITY=11\n"
        if custom_limits
        else ""
    )
    compose = Path(__file__).resolve().parents[1] / "docker-compose.yml"
    result = subprocess.run(
        [
            docker,
            "compose",
            "--project-directory",
            str(tmp_path),
            "--env-file",
            str(dotenv),
            "-f",
            str(compose),
            "config",
            "--format",
            "json",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    resolved = json.loads(result.stdout)["services"]["chatcv"]["environment"]
    assert {
        key: resolved.get(key)
        for key in (
            "CONVERSATION_MAX_TURNS",
            "CONVERSATION_IDLE_MINUTES",
            "CONVERSATION_CAPACITY",
        )
    } == {
        "CONVERSATION_MAX_TURNS": "3" if custom_limits else "20",
        "CONVERSATION_IDLE_MINUTES": "7" if custom_limits else "30",
        "CONVERSATION_CAPACITY": "11" if custom_limits else "100",
    }
