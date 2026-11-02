"""Validate Compose with synthetic settings, without starting containers."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_default_compose_only_runs_app_and_preserves_endpoint_settings(
    tmp_path: Path,
) -> None:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker CLI is not installed")
    version = subprocess.run(
        [docker, "compose", "version"], capture_output=True, check=False, timeout=10
    )
    if version.returncode:
        pytest.skip("Compose plugin is not installed")
    env_file = tmp_path / "compose.env"
    env_file.write_text(
        "LLM_PROVIDER=openai_compatible\nLLM_BASE_URL=https://chat.example/v1\n"
        "LLM_MODEL=synthetic-chat\nLLM_API_KEY=synthetic-chat-key\n"
        "EMBEDDING_PROVIDER=openai_compatible\nEMBEDDING_MODEL=synthetic-embedding\n"
        "EMBEDDING_BASE_URL=https://embedding.example/v1\nEMBEDDING_API_KEY=synthetic-embedding-key\n"
    )
    result = subprocess.run(
        [docker, "compose", "--env-file", str(env_file), "config", "--format", "json"],
        cwd=ROOT,
        env={"PATH": os.environ.get("PATH", os.defpath), "GUARDRAILS_TOKEN": ""},
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    config = json.loads(result.stdout)
    assert config["name"] == "chatcv"
    assert set(config["services"]) == {"chatcv"}
    app = config["services"]["chatcv"]
    assert Path(app["build"]["context"]) == ROOT
    assert not app.get("depends_on")
    assert not app.get("deploy")
    assert not config.get("volumes")
    assert app["environment"]["LLM_BASE_URL"] == "https://chat.example/v1"
    assert app["environment"]["LLM_MODEL"] == "synthetic-chat"
    assert app["environment"]["LLM_API_KEY"] == "synthetic-chat-key"
    assert app["environment"]["EMBEDDING_API_KEY"] == "synthetic-embedding-key"
