"""
RAG Pipeline Implementation for ChatCV

This module implements a production-ready retrieval-augmented generation pipeline
for interactive resume querying using LangChain, ChromaDB, and various LLM providers.
"""

import threading
from typing import Any

from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda

from app.config import config
from app.modules.guardrails import QueryValidator
from app.modules.model_provider import ModelProvider
from app.modules.vectorstore_provider import DocumentHandler, VectorStoreManager
from app.utils.logging_config import rag_logger

# Optional LangSmith integration
try:
    import os

    import langsmith

    # Check if LangSmith should be enabled
    langsmith_tracing = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    langsmith_api_key = os.getenv("LANGSMITH_API_KEY")

    if langsmith_tracing and langsmith_api_key:
        LANGSMITH_AVAILABLE = True
        rag_logger.info("LangSmith tracing enabled")
    else:
        LANGSMITH_AVAILABLE = False
        rag_logger.info(
            "LangSmith tracing disabled - missing API key or disabled in config"
        )

except ImportError:
    LANGSMITH_AVAILABLE = False
    rag_logger.info("LangSmith not available - continuing without tracing")

# Create a dummy decorator for when langsmith is not available or disabled
if not LANGSMITH_AVAILABLE:

    def langsmith_traceable(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    # Create a mock langsmith module
    class MockLangsmith:
        traceable = staticmethod(langsmith_traceable)


load_dotenv()


class ChatRAGPipeline:
    """Main RAG pipeline for ChatCV with thread-safe singleton pattern and modular components."""

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
        """Initialize pipeline only once"""
        if hasattr(self, "_initialized") and self._initialized:
            return

        rag_logger.info("Initializing ChatRAGPipeline singleton instance")
        self.config = config

        # Initialize modular components
        self.model_provider = ModelProvider(self.config, rag_logger)
        self.vectorstore_manager = VectorStoreManager(self.config, None, rag_logger)
        self.document_manager = DocumentHandler(self.config, rag_logger)
        self.query_validator = QueryValidator(self.config, rag_logger)

        self.qa_chain = None
        self.chat_history = None
        self._chain_lock = threading.Lock()
        self._history_lock = threading.Lock()
        self._initialized = True
        rag_logger.info("ChatRAGPipeline instance setup complete")

    def _initialize_qa_chain(self):
        """Thread-safe initialization of the QA chain."""
        if self.qa_chain is not None:
            return

        with self._chain_lock:
            if self.qa_chain is not None:
                return

            rag_logger.info("Initializing RAG QA chain")

            # Configure retrieval prompt for history awareness
            retriever_prompt = ChatPromptTemplate.from_messages(
                [
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                    (
                        "human",
                        (
                            "given the above conversation, generate a search query "
                            "to lookup content important for the conversation."
                        ),
                    ),
                ]
            )
            # Initialize vector database and retriever
            vector_database = self.vectorstore_manager.get_vectorstore()
            base_retriever = vector_database.as_retriever()
            base_retriever.search_kwargs["k"] = 8

            # Initialize language model
            language_model = self.model_provider.get_language_model()

            # Create history-aware retriever
            history_aware_retriever = create_history_aware_retriever(
                llm=language_model, retriever=base_retriever, prompt=retriever_prompt
            )

            # Use system prompt from configuration and format with candidate name
            system_prompt = self.config.chat_system_prompt.format(
                candidate_name=self.config.candidate.name
            )

            response_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{input}"),
                ]
            )

            # Create document processing chain
            document_chain = create_stuff_documents_chain(
                language_model, response_prompt
            )

            # Create complete retrieval chain
            base_qa_chain = create_retrieval_chain(
                history_aware_retriever, document_chain
            )

            # Add input and output validation
            input_validator = RunnableLambda(
                lambda x: {
                    **x,
                    "input": self.query_validator.validate_query_input(x["input"]),
                }
            )
            output_validator = RunnableLambda(
                lambda x: {
                    **x,
                    "answer": self.query_validator.validate_response_output(
                        x["answer"]
                    ),
                }
            )

            self.qa_chain = input_validator | base_qa_chain | output_validator
            rag_logger.info("RAG QA chain initialization completed")

    @langsmith.traceable(
        run_type="llm", name="Chat Completion", tags=["chatcv", "rag"], metadata={}
    )
    def get_completion(
        self, query: str, user_metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Process chat query and return response with conversation history.

        Args:
            query (str): User query string
            user_metadata (Optional[Dict]): Additional user context

        Returns:
            Dict[str, Any]: Response containing answer and sources
        """
        rag_logger.info(f"Processing chat query, length: {len(query)}")
        self._initialize_qa_chain()

        # Add user metadata to LangSmith tracing
        current_run = langsmith.get_current_run_tree()
        if current_run and user_metadata:
            current_run.metadata["user_metadata"] = user_metadata

        try:
            # Thread-safe conversation history management
            with self._history_lock:
                if self.chat_history is None:
                    self.chat_history = []

                self.chat_history.append(HumanMessage(content=query))
                current_history = self.chat_history.copy()

            # Process query through RAG pipeline
            result = self.qa_chain.invoke(
                {"input": query, "chat_history": current_history}
            )

            # Update conversation history with response
            with self._history_lock:
                self.chat_history.append(AIMessage(content=result["answer"]))

            rag_logger.info("Chat completion processed successfully")
            return {"answer": result, "sources": result.get("context", [])}

        except ValueError as validation_error:
            rag_logger.warning(f"Query validation failed: {validation_error!s}")
            return {"answer": {"answer": str(validation_error)}, "sources": []}
        except Exception:
            rag_logger.exception("Unexpected error in chat completion")
            formatted_fallback = self.config.chat_fallback_response.format(
                candidate_name=self.config.candidate.name
            )
            return {
                "answer": {"answer": formatted_fallback},
                "sources": [],
            }


def get_chat_pipeline() -> ChatRAGPipeline:
    """Get thread-safe ChatRAGPipeline instance."""
    rag_logger.info("get_chat_pipeline invoked")
    return ChatRAGPipeline()


def get_chat_completion(
    query: str, user_metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Get chat completion using thread-safe pipeline."""
    rag_logger.info("get_chat_completion invoked")
    pipeline = get_chat_pipeline()
    return pipeline.get_completion(query, user_metadata)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ChatCV RAG Pipeline Management")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Initialize vector database with document ingestion",
    )
    args = parser.parse_args()

    if args.ingest:
        rag_logger.info("Initializing vector database...")
        pipeline = get_chat_pipeline()
        pipeline.vectorstore_manager.setup_vectorstore()
        rag_logger.info("Vector database initialization complete.")
    else:
        rag_logger.info(
            "No action specified. Use --ingest to build the vector database."
        )
