from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError
from pytest import MonkeyPatch

from app.config import AppConfig, load_config

MODEL_ENV_VARS = (
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_TEMPERATURE",
    "LLM_TIMEOUT",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "OPENAI_API_KEY",
    "OLLAMA_ENDPOINT",
    "OLLAMA_TIMEOUT",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_BASE_URL",
    "EMBEDDING_API_KEY",
    "EMBEDDING_TIMEOUT",
)


@pytest.fixture
def config_directory(monkeypatch: MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("INVITE_CODES", "{}")
    for variable in MODEL_ENV_VARS:
        monkeypatch.delenv(variable, raising=False)
    directory = tmp_path / "config"
    directory.mkdir()
    (directory / "base.yml").write_text(
        "llm:\n  provider: openai\n  model: yaml-chat\n  temperature: 0.3\n"
        "  timeout: 50\nollama:\n  endpoint: http://yaml-server:11434\n  timeout: 40\n"
        "embedding:\n  provider: openai\n  model: yaml-embedding\n  chunk_size: 700\n"
    )
    return tmp_path


def test_dotenv_model_settings_override_yaml(config_directory: Path) -> None:
    (config_directory / ".env").write_text(
        "LLM_PROVIDER=ollama\nLLM_MODEL=dotenv-chat\n"
        "OLLAMA_ENDPOINT=http://dotenv-server:11434\n"
        "EMBEDDING_PROVIDER=openai\nEMBEDDING_MODEL=dotenv-embedding\n"
    )

    settings = load_config()

    assert settings.llm.provider == "ollama"
    assert settings.llm.model == "dotenv-chat"
    assert settings.ollama.endpoint == "http://dotenv-server:11434"
    assert settings.embedding.model == "dotenv-embedding"
    assert settings.embedding.provider == "openai"
    assert settings.embedding.chunk_size == 700


def test_example_environment_selects_the_shared_api_for_both_roles(
    config_directory: Path, monkeypatch: MonkeyPatch
) -> None:
    example = Path(__file__).resolve().parents[1] / ".env.example"
    (config_directory / ".env").write_text(example.read_text())
    monkeypatch.setenv("LLM_API_KEY", "example-chat-key")
    monkeypatch.setenv("EMBEDDING_API_KEY", "example-embedding-key")

    settings = load_config()

    assert settings.llm.provider == "openai_compatible"
    assert settings.embedding.provider == "openai_compatible"
    assert settings.llm.base_url == "https://api.openai.com/v1"
    assert settings.embedding.base_url == "https://api.openai.com/v1"
    assert settings.llm_api_key == SecretStr("example-chat-key")
    assert settings.embedding_api_key == SecretStr("example-embedding-key")


@pytest.mark.parametrize("variable", ["LLM_API_KEY", "EMBEDDING_API_KEY"])
def test_example_empty_keys_have_no_inline_comments_for_compose(variable: str) -> None:
    # Compose can treat a comment after an empty assignment as the value,
    # even though python-dotenv treats the same assignment as empty.
    example = Path(__file__).resolve().parents[1] / ".env.example"
    assignment = next(
        line
        for line in example.read_text().splitlines()
        if line.startswith(f"{variable}=")
    )

    assert assignment.strip() == f"{variable}="


@pytest.mark.parametrize(
    ("variable", "section", "field", "value", "expected"),
    [
        ("LLM_PROVIDER", "llm", "provider", "ollama", "ollama"),
        ("LLM_MODEL", "llm", "model", "env-chat", "env-chat"),
        ("LLM_TEMPERATURE", "llm", "temperature", "0.4", 0.4),
        ("LLM_TIMEOUT", "llm", "timeout", "19", 19),
        (
            "LLM_BASE_URL",
            "llm",
            "base_url",
            "http://chat:8000/v1",
            "http://chat:8000/v1",
        ),
        (
            "OLLAMA_ENDPOINT",
            "ollama",
            "endpoint",
            "http://env:11434",
            "http://env:11434",
        ),
        ("OLLAMA_TIMEOUT", "ollama", "timeout", "23", 23),
        ("EMBEDDING_PROVIDER", "embedding", "provider", "ollama", "ollama"),
        ("EMBEDDING_MODEL", "embedding", "model", "env-embedding", "env-embedding"),
        (
            "EMBEDDING_BASE_URL",
            "embedding",
            "base_url",
            "https://embeddings.example/v1",
            "https://embeddings.example/v1",
        ),
        ("EMBEDDING_TIMEOUT", "embedding", "timeout", "31", 31),
    ],
)
def test_process_model_settings_override_dotenv_and_yaml(
    config_directory: Path,
    monkeypatch: MonkeyPatch,
    variable: str,
    section: str,
    field: str,
    value: str,
    expected: str | float,
) -> None:
    (config_directory / ".env").write_text("LLM_MODEL=dotenv-chat\nOLLAMA_TIMEOUT=12\n")
    monkeypatch.setenv(variable, value)

    settings: AppConfig = load_config()

    assert getattr(getattr(settings, section), field) == expected


def test_blank_model_settings_keep_yaml_defaults(
    config_directory: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "")
    monkeypatch.setenv("OLLAMA_ENDPOINT", "")

    settings = load_config()

    assert settings.llm.provider == "openai"
    assert settings.ollama.endpoint == "http://yaml-server:11434"


def test_dotenv_can_configure_compatible_endpoints_and_omit_temperature(
    config_directory: Path,
) -> None:
    (config_directory / ".env").write_text(
        "LLM_PROVIDER=openai_compatible\nLLM_BASE_URL=http://chat:8000/v1\n"
        "LLM_TEMPERATURE=null\nLLM_API_KEY=dotenv-chat-key\n"
        "EMBEDDING_PROVIDER=openai_compatible\n"
        "EMBEDDING_BASE_URL=http://embedding:8001/v1\n"
        "EMBEDDING_API_KEY=dotenv-embedding-key\nEMBEDDING_TIMEOUT=21\n"
    )

    settings = load_config()

    assert settings.llm.provider == "openai_compatible"
    assert settings.llm.base_url == "http://chat:8000/v1"
    assert settings.llm.temperature is None
    assert settings.llm_api_key == SecretStr("dotenv-chat-key")
    assert settings.embedding.provider == "openai_compatible"
    assert settings.embedding.base_url == "http://embedding:8001/v1"
    assert settings.embedding_api_key == SecretStr("dotenv-embedding-key")
    assert settings.embedding.timeout == 21


@pytest.mark.parametrize(
    "variable", ["OPENAI_API_KEY", "LLM_API_KEY", "EMBEDDING_API_KEY"]
)
def test_process_api_key_overrides_dotenv_and_is_redacted(
    config_directory: Path, monkeypatch: MonkeyPatch, variable: str
) -> None:
    (config_directory / ".env").write_text(f"{variable}=dotenv-secret\n")
    monkeypatch.setenv(variable, "process-secret")

    settings = load_config()

    assert getattr(settings, variable.lower()) == SecretStr("process-secret")
    assert "process-secret" not in repr(settings)
    assert "dotenv-secret" not in repr(settings)


@pytest.mark.parametrize(
    "variable", ["LLM_TIMEOUT", "OLLAMA_TIMEOUT", "EMBEDDING_TIMEOUT"]
)
def test_request_timeouts_must_be_positive(
    config_directory: Path, monkeypatch: MonkeyPatch, variable: str
) -> None:
    monkeypatch.setenv(variable, "0")

    with pytest.raises(ValidationError, match="greater than 0"):
        load_config()
