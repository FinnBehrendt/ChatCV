import logging
import os
import re

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import AppConfig
from app.modules.guardrails import InputValidator
from app.modules.model_provider import ModelProvider


class VectorStoreManager:
    """Manages vector database setup and retrieval."""

    def __init__(
        self,
        config: AppConfig,
        model_provider: ModelProvider | None,
        logger: logging.Logger,
    ) -> None:
        """
        Initialize the VectorStoreManager.

        Args:
            config (AppConfig): Application configuration object.
            model_provider (Optional[ModelProvider]): Model provider instance or None.
            logger (logging.Logger): Logger instance for logging.
        """
        self.config = config
        self.model_provider = ModelProvider(config, logger)
        self.logger = logger

    def setup_vectorstore(self) -> Chroma:
        """
        Initialize and populate the vector database with resume documents.

        Loads CV and About Me documents, splits them into chunks, and creates a vector database.

        Returns:
            Chroma: Configured vector database instance.
        """
        self.logger.info("Setting up vectorstore with document ingestion")

        try:
            # Load primary CV document
            cv_loader = PyPDFLoader(self.config.data.cv_path)
            cv_documents = cv_loader.load()
            self.logger.info(f"Loaded {len(cv_documents)} CV documents")
            for doc in cv_documents:
                doc.metadata["source"] = "cv"
        except Exception:
            self.logger.exception("Failed to load CV documents")
            cv_documents = []

        try:
            # Load additional context document
            about_loader = TextLoader(self.config.data.about_me_path)
            about_documents = about_loader.load()
            self.logger.info(f"Loaded {len(about_documents)} About Me documents")
            for doc in about_documents:
                doc.metadata["source"] = "about_me"
        except Exception:
            self.logger.exception("Failed to load About Me documents")
            about_documents = []

        # Combine document sources
        all_documents = cv_documents + about_documents
        self.logger.info(f"Total documents before splitting: {len(all_documents)}")

        # Configure document chunking strategy from config
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        document_chunks = text_splitter.split_documents(all_documents)
        self.logger.info(
            f"Total document chunks after splitting: {len(document_chunks)}"
        )

        # Initialize vector database
        embedding_function = self.model_provider.get_embedding_model()
        vector_database = Chroma.from_documents(
            document_chunks,
            embedding_function,
            persist_directory=self.config.data.vector_db_path,
        )

        self.logger.info(
            f"Vectorstore initialized with {len(document_chunks)} document chunks"
        )
        return vector_database

    def get_vectorstore(self) -> Chroma:
        """
        Retrieve existing vector database or create new one if not found.

        Returns:
            Chroma: Vector database instance.
        """
        vector_db_path = self.config.data.vector_db_path

        # Check if vector database exists and contains data
        if not os.path.exists(vector_db_path) or not self._is_vectorstore_populated(
            vector_db_path
        ):
            self.logger.warning(
                f"Vector database not found or empty at {vector_db_path}. Initializing..."
            )
            return self.setup_vectorstore()

        self.logger.info(f"Loading existing vector database from {vector_db_path}")
        embedding_function = self.model_provider.get_embedding_model()
        return Chroma(
            persist_directory=vector_db_path, embedding_function=embedding_function
        )

    def _is_vectorstore_populated(self, vector_db_path: str) -> bool:
        """
        Check if the vector database directory contains actual data.

        Args:
            vector_db_path (str): Path to the vector database directory.

        Returns:
            bool: True if the database appears to be populated, False otherwise.
        """
        try:
            # Check if the directory is empty
            if not os.listdir(vector_db_path):
                return False

            # Check for Chroma-specific files that indicate a populated database
            chroma_files = ["chroma.sqlite3", "index"]
            has_chroma_files = any(
                os.path.exists(os.path.join(vector_db_path, file))
                for file in chroma_files
            )

            if not has_chroma_files:
                return False

            # Try to load the database to verify it's valid and has content
            embedding_function = self.model_provider.get_embedding_model()
            temp_db = Chroma(
                persist_directory=vector_db_path, embedding_function=embedding_function
            )

            # Check if database actually contains documents
            collection = temp_db._collection
            if collection.count() == 0:
                self.logger.warning("Vector database exists but contains no documents")
                return False

            self.logger.info(f"Vector database contains {collection.count()} documents")
            return True

        except Exception:
            self.logger.warning(
                "Error checking vector database population", exc_info=True
            )
            return False


class DocumentHandler:
    """Handles document content extraction and text processing."""

    def __init__(self, config: AppConfig, logger: logging.Logger) -> None:
        """
        Initialize the DocumentHandler.

        Args:
            config (AppConfig): Application configuration object.
            logger (logging.Logger): Logger instance for logging.
        """
        self.config = config
        self.logger = logger
        self.validator = InputValidator(config, logger)

    def get_cv_content(self) -> str:
        """
        Extract complete CV content as a single concatenated string.

        Returns:
            str: Concatenated CV content.
        """
        try:
            cv_loader = PyPDFLoader(self.config.data.cv_path)
            cv_documents = cv_loader.load()
            self.logger.info(f"Loaded {len(cv_documents)} CV document pages")
        except Exception:
            self.logger.exception("Failed to load CV document")
            return ""

        cv_content = [doc.page_content for doc in cv_documents if doc.page_content]

        if not cv_content:
            self.logger.warning("CV document appears to be empty or unreadable.")

        return " ".join(cv_content)

    def process_text(self, text: str) -> Document:
        """
        Process plain text job description into a Document object.

        Args:
            text (str): Raw job description text.

        Returns:
            Document: Processed job description document.

        Raises:
            ValueError: If text is empty or too short.
        """
        min_length = self.config.security.min_job_text_length

        if not text or len(text.strip()) < min_length:
            error_msg = f"Job description text must be at least {min_length} characters"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # validate if text contains disallowed HTML tags or event handlers
        self.validator.validate_job_text(text)

        # Clean and normalize text
        cleaned_text = re.sub(r"\s+", " ", text.strip())

        self.logger.info(f"Processed job description text, length: {len(cleaned_text)}")
        self.logger.debug("Job description metadata source: text_input")

        return Document(
            page_content=cleaned_text,
            metadata={"source": "text_input", "type": "job_description"},
        )

    def load_documents(self) -> list[Document]:
        """
        Load and prepare CV and About Me documents for summarization.

        Returns:
            List[Document]: Combined and annotated document list.
        """
        self.logger.info("Loading documents for summary generation")

        try:
            # Load CV (PDF)
            cv_loader = PyPDFLoader(self.config.data.cv_path)
            docs_cv = cv_loader.load()
            self.logger.info(f"Loaded {len(docs_cv)} CV documents")
            for doc in docs_cv:
                doc.metadata["source"] = "cv"
        except Exception:
            self.logger.exception("Failed to load CV documents")
            docs_cv = []

        docs_about = []
        about_me_path = self.config.data.about_me_path
        if os.path.exists(about_me_path):
            try:
                about_loader = TextLoader(about_me_path)
                docs_about = about_loader.load()
                self.logger.info(f"Loaded {len(docs_about)} About Me documents")
                for doc in docs_about:
                    doc.metadata["source"] = "about_me"
            except Exception:
                self.logger.exception("Failed to load About Me documents")
                docs_about = []
        else:
            self.logger.warning(f"About Me document not found at path: {about_me_path}")

        total_docs = len(docs_cv) + len(docs_about)
        self.logger.info(
            f"Loaded {total_docs} documents ({len(docs_cv)} CV, {len(docs_about)} about)"
        )

        return docs_cv + docs_about
