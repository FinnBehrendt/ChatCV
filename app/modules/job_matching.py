"""
Job Description Matching Pipeline for ChatCV

This module implements job description analysis and candidate matching
using RAG-based assessment to evaluate candidate fit.
"""

import threading
from typing import Any

import langsmith
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.config import config
from app.modules.guardrails import InputValidator
from app.modules.model_provider import ModelProvider
from app.modules.vectorstore_provider import DocumentHandler, VectorStoreManager
from app.utils.logging_config import job_matching_logger

load_dotenv()


class JobMatchingAnalyzer:
    """Performs job matching analysis using RAG pipeline"""

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
        """Initialize analyzer only once"""
        if hasattr(self, "_initialized") and self._initialized:
            return

        job_matching_logger.info("Initializing JobMatchingAnalyzer singleton instance")
        self.model_provider = ModelProvider(config, job_matching_logger)
        self.llm = self.model_provider.get_language_model()
        self.validator = InputValidator(config, job_matching_logger)
        self.vectorstore_manager = VectorStoreManager(config, None, job_matching_logger)
        self.matching_chain = None
        self._chain_lock = threading.Lock()
        self._initialized = True
        job_matching_logger.info("JobMatchingAnalyzer initialization completed")

    def _initialize_matching_chain(self):
        """Thread-safe initialization of the job matching analysis chain"""
        if self.matching_chain is not None:
            job_matching_logger.debug(
                "Matching chain already initialized, skipping initialization"
            )
            return

        with self._chain_lock:
            if self.matching_chain is not None:
                job_matching_logger.debug(
                    "Matching chain already initialized inside lock, skipping initialization"
                )
                return

            job_matching_logger.info("Initializing job matching analysis chain")

            # Use configurable job matching prompt
            system_prompt = config.job_matching_system_prompt

            analysis_prompt = config.job_matching_analysis_prompt.format(
                candidate_name=config.candidate.name
            )

            # Combine system and analysis prompts
            full_prompt = f"{system_prompt}\n\n{analysis_prompt}"

            prompt = ChatPromptTemplate.from_template(full_prompt)
            self.matching_chain = create_stuff_documents_chain(self.llm, prompt)
            job_matching_logger.info(
                "Job matching analysis chain initialization completed"
            )

    @langsmith.traceable(
        run_type="llm",
        name="Job Matching Analysis",
        tags=["job_matching", "assessment"],
        metadata={},
    )
    def analyze_job_match(
        self, job_description: Document, user_metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Analyze job match using RAG-based assessment.

        Args:
            job_description (Document): The job description document.
            user_metadata (Optional[Dict[str, Any]]): Optional user metadata for tracing.

        Returns:
            Dict[str, Any]: Analysis results including matching details and metadata.
        """
        job_matching_logger.info("Starting job matching analysis")
        self._initialize_matching_chain()

        # Add metadata to LangSmith tracing
        current_run = langsmith.get_current_run_tree()
        if current_run and user_metadata:
            current_run.metadata["user_metadata"] = user_metadata
            current_run.metadata["job_source"] = job_description.metadata.get(
                "source", "unknown"
            )

        try:
            # Retrieve relevant candidate information
            vectorstore = self.vectorstore_manager.get_vectorstore()
            retrieval_k = config.vectorstore.retrieval_k
            retriever = vectorstore.as_retriever(search_kwargs={"k": retrieval_k})
            relevant_docs = retriever.invoke(job_description.page_content)
            job_matching_logger.info(
                f"Retrieved {len(relevant_docs)} relevant documents from vectorstore"
            )

            # Create job description document
            job_doc = Document(
                page_content=f"JOB DESCRIPTION:\n{job_description.page_content}",
                metadata={"source": "job_description", "type": "job_requirements"},
            )

            # Validate job description content
            job_matching_logger.info("Validating job description content")
            self.validator.validate_job_text(job_doc.page_content)
            job_matching_logger.info("Job description validation completed")

            # Combine all documents for context
            all_docs = [job_doc] + relevant_docs

            job_matching_logger.info(
                f"Analyzing job match with {len(relevant_docs)} relevant CV sections"
            )

            # Perform analysis
            analysis_result = self.matching_chain.invoke({"context": all_docs})

            job_matching_logger.info("Job matching analysis completed successfully")

            return {
                "analysis": analysis_result,
                "job_source": job_description.metadata.get("source", "unknown"),
                "match_timestamp": str(current_run.start_time) if current_run else None,
                "relevant_sections": [
                    doc.metadata.get("source", "unknown") for doc in relevant_docs
                ],
            }

        except Exception as e:
            job_matching_logger.error(
                f"Job matching analysis error ({e.__class__.__name__}): {e!s}"
            )
            return {
                "analysis": "Unable to complete job matching analysis. Please try again.",
                "error": str(e),
                "job_source": job_description.metadata.get("source", "unknown"),
            }


# Thread-safe singleton access
def get_job_analyzer() -> JobMatchingAnalyzer:
    """Get thread-safe job matching analyzer instance.

    Returns:
        JobMatchingAnalyzer: Singleton instance of job matching analyzer.
    """
    return JobMatchingAnalyzer()


def process_job_description(text: str) -> Document:
    """Process job description from text input only.

    Args:
        text (str): Raw job description text.

    Raises:
        ValueError: If the input text is empty.

    Returns:
        Document: Processed job description document.
    """
    if not text:
        raise ValueError("Job description text is required")

    handler = DocumentHandler(config, job_matching_logger)
    return handler.process_text(text)


def analyze_job_match(
    job_description: Document, user_metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Analyze job match for given job description.

    Args:
        job_description (Document): The job description document.
        user_metadata (Optional[Dict[str, Any]]): Optional user metadata for tracing.

    Returns:
        Dict[str, Any]: Analysis results including matching details and metadata.
    """
    analyzer = get_job_analyzer()
    return analyzer.analyze_job_match(job_description, user_metadata)
