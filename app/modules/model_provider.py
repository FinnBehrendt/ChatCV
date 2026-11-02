import logging
from urllib.parse import urlsplit

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from app.config import AppConfig


class ModelProvider:
    """
    A provider class responsible for initializing and returning language and embedding models
    based on the given application configuration.

    Attributes:
        config (AppConfig): The application configuration containing model settings.
        logger (logging.Logger): Logger instance for logging information and debug messages.
    """

    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        """
        Initialize the ModelProvider with configuration and logger.

        Args:
            config (AppConfig): Configuration object with model parameters.
            logger (logging.Logger): Logger for recording operational messages.
        """
        self.config = config
        self.logger = logger

    @staticmethod
    def _compatible_endpoint(value: str | None, setting: str) -> str:
        """Require an explicit endpoint without credentials in its URL."""
        message = (
            f"{setting} must be an HTTP(S) URL without credentials, query or fragment"
        )
        try:
            url = urlsplit((value or "").strip())
            valid = (
                url.scheme in {"http", "https"}
                and url.hostname
                and not (url.username or url.password or url.query or url.fragment)
            )
        except ValueError:
            raise ValueError(message) from None
        if not valid:
            raise ValueError(message)
        return (value or "").strip().rstrip("/")

    def get_language_model(self) -> BaseChatModel:
        """
        Initialize and return the language model instance based on the configuration.

        Selects the provider explicitly.

        Returns:
            BaseChatModel: A chat client with LangChain's common interface.
        """

        self.logger.info(
            f"Starting initialization of language model '{self.config.llm.model}'."
        )
        provider = self.config.llm.provider.strip().lower()
        if provider == "ollama":
            return ChatOllama(
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
                base_url=self.config.ollama.endpoint,
                client_kwargs={"timeout": self.config.ollama.timeout},
            )
        if provider == "openai":
            return ChatOpenAI(
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
                timeout=self.config.llm.timeout,
                api_key=self.config.llm_api_key or self.config.openai_api_key,
                base_url="https://api.openai.com/v1",
            )
        if provider == "openai_compatible":
            endpoint = self._compatible_endpoint(
                self.config.llm.base_url, "LLM_BASE_URL"
            )
            return ChatOpenAI(
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
                timeout=self.config.llm.timeout,
                base_url=endpoint,
                # Never let the SDK pick up an unrelated OPENAI_API_KEY.
                api_key=self.config.llm_api_key or SecretStr("not-required"),
                use_responses_api=False,
                stream_usage=False,
            )
        raise ValueError(f"Unsupported chat provider: {provider!r}")

    def get_embedding_model(self) -> Embeddings:
        """
        Initialize and return the embedding model instance based on the configuration.

        Selects embeddings independently from the chat provider and model name.

        Returns:
            Embeddings: An embedding client independent of the chat provider.
        """
        provider = self.config.embedding.provider.strip().lower()
        self.logger.info(
            f"Starting initialization of embedding model '{self.config.embedding.model}'."
        )
        if provider == "ollama":
            return OllamaEmbeddings(
                model=self.config.embedding.model,
                base_url=self.config.ollama.endpoint,
                client_kwargs={"timeout": self.config.ollama.timeout},
            )
        if provider == "openai":
            return OpenAIEmbeddings(
                model=self.config.embedding.model,
                api_key=self.config.embedding_api_key or self.config.openai_api_key,
                base_url="https://api.openai.com/v1",
                timeout=self.config.embedding.timeout,
            )
        if provider == "openai_compatible":
            endpoint = self._compatible_endpoint(
                self.config.embedding.base_url, "EMBEDDING_BASE_URL"
            )
            return OpenAIEmbeddings(
                model=self.config.embedding.model,
                base_url=endpoint,
                api_key=self.config.embedding_api_key or SecretStr("not-required"),
                timeout=self.config.embedding.timeout,
                # Other servers may accept text but not OpenAI-specific token IDs.
                check_embedding_ctx_length=False,
                model_kwargs={"encoding_format": "float"},
            )
        raise ValueError(f"Unsupported embedding provider: {provider!r}")
