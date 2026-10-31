import os
import subprocess
from pathlib import Path

import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "ollama" / "prepare-models.sh"


def run_preparation(
    tmp_path: Path,
    *,
    available_models: str = "",
    embedding_base_url: str = "https://embedding.example/v1",
    pull_exit_code: int = 0,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Run the real shell script against a fake CLI, with no server or downloads."""
    cli = tmp_path / "ollama"
    cli.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "show" ]; then\n'
        '    case " $FAKE_AVAILABLE_MODELS " in\n'
        '        *" $2 "*) exit 0 ;;\n'
        "        *) exit 1 ;;\n"
        "    esac\n"
        "fi\n"
        'if [ "$1" = "pull" ]; then\n'
        '    printf "%s\\n" "$2" >> "$FAKE_PULL_LOG"\n'
        '    exit "$FAKE_PULL_EXIT_CODE"\n'
        "fi\n"
        "exit 2\n"
    )
    cli.chmod(0o700)
    pull_log = tmp_path / "pulls.txt"
    result = subprocess.run(
        ["sh", str(SCRIPT)],
        env={
            "PATH": f"{tmp_path}{os.pathsep}{os.defpath}",
            "OLLAMA_CHAT_MODEL": "test-chat",
            "EMBEDDING_PROVIDER": "openai_compatible",
            "EMBEDDING_BASE_URL": embedding_base_url,
            "EMBEDDING_MODEL": "test-embedding",
            "FAKE_AVAILABLE_MODELS": available_models,
            "FAKE_PULL_LOG": str(pull_log),
            "FAKE_PULL_EXIT_CODE": str(pull_exit_code),
        },
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    pulls = pull_log.read_text().splitlines() if pull_log.exists() else []
    return result, pulls


def test_missing_chat_model_is_downloaded(tmp_path: Path) -> None:
    result, pulls = run_preparation(tmp_path)

    assert result.returncode == 0
    assert pulls == ["test-chat"]


def test_existing_chat_model_is_reused_without_download(tmp_path: Path) -> None:
    result, pulls = run_preparation(tmp_path, available_models="test-chat")

    assert result.returncode == 0
    assert pulls == []


def test_local_embeddings_are_prepared_after_chat(tmp_path: Path) -> None:
    result, pulls = run_preparation(
        tmp_path, embedding_base_url="http://ollama:11434/v1/"
    )

    assert result.returncode == 0
    assert pulls == ["test-chat", "test-embedding"]


def test_failed_download_stops_preparation(tmp_path: Path) -> None:
    result, pulls = run_preparation(
        tmp_path, pull_exit_code=7, embedding_base_url="http://ollama:11434/v1"
    )

    assert result.returncode == 7
    assert pulls == ["test-chat"]


def test_base_compose_only_starts_the_application() -> None:
    root = SCRIPT.parents[1]
    base = yaml.safe_load((root / "docker-compose.yml").read_text())

    assert set(base["services"]) == {"chatcv"}
    assert not base.get("volumes")
    assert not base["services"]["chatcv"].get("depends_on")


def test_local_override_owns_model_services_and_startup_order() -> None:
    root = SCRIPT.parents[1]
    local = yaml.safe_load((root / "docker-compose.ollama.yml").read_text())
    services = local["services"]

    assert set(services) == {"chatcv", "ollama", "ollama-init"}
    assert all(not service.get("profiles") for service in services.values())
    assert "ollama_data" in local["volumes"]
    assert "ollama_data:/root/.ollama" in services["ollama"]["volumes"]
    assert (
        services["ollama-init"]["depends_on"]["ollama"]["condition"]
        == "service_healthy"
    )
    assert (
        services["chatcv"]["depends_on"]["ollama-init"]["condition"]
        == "service_completed_successfully"
    )
