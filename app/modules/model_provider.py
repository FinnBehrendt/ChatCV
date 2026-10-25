import logging

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import AppConfig


class ModelProvider:
    """
    A provider class responsible for initializing and returning language and embedding models
    based on the given application configuration.

    Attributes:
        config (AppConfig): The application configuration containing model settings.
        logger (logging.Logger): Logger instance for logging information and debug messages.
    """

    def __init__(self, config: AppConfig, logger: logging.Logger):
        """
        Initialize the ModelProvider with configuration and logger.

        Args:
            config (AppConfig): Configuration object with model parameters.
            logger (logging.Logger): Logger for recording operational messages.
        """
        self.config = config
        self.logger = logger

    def get_language_model(self) -> ChatOpenAI | ChatOllama:
        """
        Initialize and return the language model instance based on the configuration.

        Detects whether to use Ollama or OpenAI language model by inspecting the model name.

        Returns:
            Union[ChatOpenAI, ChatOllama]: An instance of the configured language model.
        """

        self.logger.info(
            f"Starting initialization of language model '{self.config.llm.model}'."
        )
        if "ollama" in str(self.config.llm.provider).lower():
            self.logger.info(
                "Detected Ollama language model source based on model name."
            )
            return ChatOllama(
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
            )
        else:
            self.logger.info("Detected OpenAI language model source as default.")
            return ChatOpenAI(
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
                timeout=self.config.llm.timeout,
            )

    def get_embedding_model(self) -> OpenAIEmbeddings | OllamaEmbeddings:
        """
        Initialize and return the embedding model instance based on the configuration.

        Detects whether to use Ollama or OpenAI embeddings by inspecting the model name.

        Returns:
            Union[OpenAIEmbeddings, OllamaEmbeddings]: An instance of the configured embedding model.
        """
        model_name = str(self.config.embedding.model).lower()
        self.logger.info(
            f"Starting initialization of embedding model '{self.config.embedding.model}'."
        )
        if "ollama" in model_name:
            self.logger.info(
                "Detected Ollama embedding model source based on model name."
            )
            return OllamaEmbeddings(
                model=self.config.embedding.model, base_url=self.config.ollama.endpoint
            )
        else:
            self.logger.info("Detected OpenAI embedding model source as default.")
            return OpenAIEmbeddings(model=self.config.embedding.model)
