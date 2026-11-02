import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings


class LLMConfig(BaseModel):
    """Configuration for the Large Language Model (LLM) settings."""

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float | None = 0.1
    timeout: int = Field(default=60, gt=0)
    base_url: str | None = None

    @field_validator("temperature", mode="before")
    @classmethod
    def allow_omitted_temperature(cls, value: object) -> object:
        if isinstance(value, str) and value.strip().lower() in {"null", "none"}:
            return None
        return value


class OllamaConfig(BaseModel):
    """Configuration for the Ollama model endpoint and parameters."""

    endpoint: str = "http://localhost:11434"
    models: str = "llama3.2"
    timeout: int = Field(default=120, gt=0)


class EmbeddingConfig(BaseModel):
    """Configuration for embedding model parameters."""

    provider: str = "openai"
    model: str = "text-embedding-3-small"
    base_url: str | None = None
    timeout: int = Field(default=60, gt=0)
    chunk_size: int = 1000
    chunk_overlap: int = 200


class VectorStoreConfig(BaseModel):
    """Configuration for vector store settings."""

    provider: str = "chroma"
    persist_directory: str = "data/vector_db"
    collection_name: str = "chatcv"
    retrieval_k: int = 8


class RateLimitConfig(BaseModel):
    """Configuration for rate limiting API requests."""

    enabled: bool = True
    requests_per_minute: int = 20
    burst_limit: int = 5
    rate_limit_ms: int = 3000


class SecurityConfig(BaseModel):
    """Configuration for security-related settings."""

    session_timeout_hours: int = 24
    max_query_length: int = 1000
    max_job_text_length: int = 5000
    min_job_text_length: int = 50
    allowed_origins: list[str] = ["*"]
    secure_cookies: bool = True


class LoggingConfig(BaseModel):
    """Configuration for logging behavior and file management."""

    level: str = "INFO"
    file_path: str = "logs/chatcv.log"
    max_file_size_mb: int = 100
    backup_count: int = 5
    analytics_enabled: bool = True


class DataPaths(BaseModel):
    """Configuration for various data file paths used by the application."""

    cv_path: str = "data/CV_Demo.pdf"
    about_me_path: str = "data/about_me.md"
    vector_db_path: str = "data/vector_db"
    analytics_log_path: str = "data/analytics.log"


class CandidateConfig(BaseModel):
    """Configuration for candidate information displayed in the application."""

    name: str = "Finn Behrendt"
    email: str = "finn.behrendt@example.com"
    linkedin: str = "https://linkedin.com/in/finn-behrendt"
    github: str = "https://github.com/finnbehrendt"


class AppConfig(BaseSettings):
    """
    Main application configuration class that aggregates all sub-configurations.

    This includes environment settings, API keys, model configurations,
    rate limiting, security, logging, data paths, and invite codes.
    """

    environment: str = Field(default="development")
    debug: bool = False
    openai_api_key: SecretStr | None = Field(
        default=None, alias="OPENAI_API_KEY", repr=False
    )
    llm_api_key: SecretStr | None = Field(default=None, alias="LLM_API_KEY", repr=False)
    embedding_api_key: SecretStr | None = Field(
        default=None, alias="EMBEDDING_API_KEY", repr=False
    )
    langsmith_api_key: str | None = Field(default=None, alias="LANGSMITH_API_KEY")

    llm: LLMConfig = LLMConfig()
    ollama: OllamaConfig = OllamaConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    vectorstore: VectorStoreConfig = VectorStoreConfig()
    rate_limit: RateLimitConfig = RateLimitConfig()
    security: SecurityConfig = SecurityConfig()
    logging: LoggingConfig = LoggingConfig()
    data: DataPaths = DataPaths()
    candidate: CandidateConfig = CandidateConfig()
    invite_codes: dict[str, Any] = {}

    # Environment variables for invite codes
    invite_codes_env: str | None = Field(default=None, alias="INVITE_CODES")

    # Chat fallback response for guardrails
    chat_fallback_response: str = (
        "I'm sorry, but I cannot provide a response to that query."
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._load_invite_codes_from_env()

    def _load_invite_codes_from_env(self):
        """Load invite codes from environment variables."""
        import json

        # If INVITE_CODES environment variable is set, use it
        if self.invite_codes_env:
            try:
                env_codes = json.loads(self.invite_codes_env)
                self.invite_codes.update(env_codes)
            except json.JSONDecodeError as e:
                print(
                    f"Warning: Invalid JSON in INVITE_CODES environment variable: {e}"
                )


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge two dictionaries.

    Values from the override dictionary take precedence over those in the base dictionary.
    Nested dictionaries are merged recursively.

    Args:
        base (Dict[str, Any]): The base dictionary to be merged into.
        override (Dict[str, Any]): The dictionary with overriding values.

    Returns:
        Dict[str, Any]: The merged dictionary.
    """
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config() -> AppConfig:
    """
    Load and merge configuration from YAML files and environment variables.

    Loads the base configuration from 'config/base.yml' and environment-specific overrides
    from 'config/{ENVIRONMENT}.yml'. Then overlays these settings with environment variables
    and returns a fully constructed AppConfig instance.

    Returns:
        AppConfig: The loaded and merged application configuration.
    """
    env = os.getenv("ENVIRONMENT", "development")
    base = Path("config/base.yml")
    override = Path(f"config/{env}.yml")

    merged_data: dict[str, Any] = {}

    if base.exists():
        with open(base, "r") as f:
            merged_data = yaml.safe_load(f) or {}

    if override.exists():
        with open(override, "r") as f:
            override_data = yaml.safe_load(f) or {}
            merged_data = deep_merge(merged_data, override_data)

    # Flat model variables documented in .env.example must override nested YAML.
    # Reading .env does not export credentials into the process environment.
    model_environment = {**dotenv_values(".env"), **os.environ}
    model_fields = {
        "LLM_PROVIDER": ("llm", "provider"),
        "LLM_MODEL": ("llm", "model"),
        "LLM_TEMPERATURE": ("llm", "temperature"),
        "LLM_TIMEOUT": ("llm", "timeout"),
        "LLM_BASE_URL": ("llm", "base_url"),
        "OLLAMA_ENDPOINT": ("ollama", "endpoint"),
        "OLLAMA_TIMEOUT": ("ollama", "timeout"),
        "EMBEDDING_PROVIDER": ("embedding", "provider"),
        "EMBEDDING_MODEL": ("embedding", "model"),
        "EMBEDDING_BASE_URL": ("embedding", "base_url"),
        "EMBEDDING_TIMEOUT": ("embedding", "timeout"),
    }
    for variable, (section, field) in model_fields.items():
        value = model_environment.get(variable)
        if value is not None and value.strip():
            merged_data.setdefault(section, {})[field] = value.strip()

    # Pydantic validates the resulting settings, including numeric values.
    return AppConfig(**merged_data)


config = load_config()
