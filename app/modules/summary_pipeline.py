"""
Summary Generation Pipeline for ChatCV

This module implements professional summary generation functionality
using document summarization techniques with LangChain and LLM providers.
"""

import threading
from typing import Any

import langsmith
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate

from app.config import config
from app.modules.model_provider import ModelProvider
from app.modules.vectorstore_provider import DocumentHandler
from app.utils.logging_config import summary_logger

load_dotenv()


class SummaryGenerator:
    """Generates professional summaries using LLM with thread-safe singleton pattern"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Thread-safe singleton implementation"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize generator only once"""
        if hasattr(self, "_initialized") and self._initialized:
            return

        summary_logger.info("Initializing SummaryGenerator singleton instance")
        self.model_provider = ModelProvider(config, summary_logger)
        self.llm = self.model_provider.get_language_model()
        self.summary_chain = None
        self._chain_lock = threading.Lock()
        self._initialized = True
        summary_logger.info("SummaryGenerator initialization completed")

    def _initialize_summary_chain(self):
        """Thread-safe initialization of the summary generation chain"""
        if self.summary_chain is not None:
            summary_logger.info(
                "Summary chain initialization skipped: already initialized"
            )
            return

        with self._chain_lock:
            if self.summary_chain is not None:
                summary_logger.info(
                    "Summary chain initialization skipped inside lock: already initialized"
                )
                return

            summary_logger.info("Initializing summary generation chain")

            # Use a comprehensive prompt template for professional summary
            prompt_template = config.summary_prompt_template.format(
                candidate_name=config.candidate.name
            )

            prompt = PromptTemplate.from_template(prompt_template)
            self.summary_chain = create_stuff_documents_chain(
                llm=self.llm, prompt=prompt
            )
            summary_logger.info("Summary chain initialization completed")

    @langsmith.traceable(
        run_type="llm",
        name="Summary Generation",
        tags=["summary", "generation"],
        metadata={},
    )
    def generate_summary(
        self, style: str = "bullet", user_metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Generate professional summary"""
        summary_logger.info(f"Starting summary generation with style: {style}")
        self._initialize_summary_chain()

        # Add metadata to LangSmith tracing
        current_run = langsmith.get_current_run_tree()
        if current_run and user_metadata:
            current_run.metadata["user_metadata"] = user_metadata
            current_run.metadata["summary_style"] = style

        try:
            # Load documents using DocumentHandler
            handler = DocumentHandler(config, summary_logger)
            documents = handler.load_documents()

            # Generate summary
            result = self.summary_chain.invoke({"context": documents})
            summary_logger.info(
                f"Summary generated successfully, length: {len(result)} characters, documents used: {len(documents)}"
            )

            return {
                "summary_md": result,
                "skills": [],
                "style": style,
                "timestamp": str(current_run.start_time) if current_run else None,
            }

        except Exception as e:
            summary_logger.error(
                f"Summary generation failed ({e.__class__.__name__}): {e!s}"
            )
            return {
                "summary_md": "Unable to generate summary. Please try again.",
                "skills": [],
                "error": str(e),
            }


# Thread-safe singleton access
def get_summary_generator() -> SummaryGenerator:
    """Get thread-safe summary generator instance"""
    return SummaryGenerator()


# Backward compatibility functions
def get_auto_summary(
    style: str = "bullet", user_metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Backward compatible summary generation function"""
    generator = get_summary_generator()
    return generator.generate_summary(style, user_metadata)


def get_llm_model() -> Any:
    """Legacy function for getting LLM model"""
    generator = get_summary_generator()
    return generator.model_provider.get_language_model()


def get_embedding_model() -> Any:
    """Legacy function for getting embedding model"""
    generator = get_summary_generator()
    return generator.model_provider.get_embedding_model()


def init_summary_chain(user_metadata: dict[str, Any] | None = None) -> None:
    """Legacy function for initializing summary chain"""
    generator = get_summary_generator()
    generator._initialize_summary_chain()
