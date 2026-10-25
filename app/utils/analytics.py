import json
import os
from datetime import datetime
from typing import Any

from app.config import config
from app.utils.logging_config import analytics_logger

# Optional LangSmith integration
try:
    from langsmith import Client as LangSmithClient

    LANGSMITH_AVAILABLE = True
    analytics_logger.info("LangSmith analytics available")
except ImportError:
    LANGSMITH_AVAILABLE = False
    LangSmithClient = None  # type: ignore
    analytics_logger.info("LangSmith not available - basic analytics only")


class AdvancedAnalytics:
    def __init__(self) -> None:
        """
        Initialize the AdvancedAnalytics instance.

        Sets up configuration, initializes variables for LangSmith client usage,
        and ensures the analytics log directory exists.
        """
        self.config = config
        self.langsmith_client = None
        self.mlflow_enabled = False
        self.analytics_file = self.config.data.analytics_log_path
        self.active_chat_runs: dict[str, dict[str, Any]] = {}

        # Ensure analytics directory exists
        os.makedirs(os.path.dirname(self.analytics_file), exist_ok=True)

        # Initialize LangSmith client if enabled and configured
        langsmith_enabled = (
            LANGSMITH_AVAILABLE
            and self.config.langsmith_api_key
            and os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
        )

        if langsmith_enabled:
            try:
                self.langsmith_client = LangSmithClient()
                analytics_logger.info("LangSmith client initialized successfully")
            except Exception as e:
                analytics_logger.warning(f"Failed to initialize LangSmith client: {e}")
                self.langsmith_client = None
        else:
            self.langsmith_client = None
            if not LANGSMITH_AVAILABLE:
                analytics_logger.info("LangSmith not available")
            elif not self.config.langsmith_api_key:
                analytics_logger.info("LangSmith disabled - no API key")
            else:
                analytics_logger.info("LangSmith disabled - LANGSMITH_TRACING=false")

        analytics_logger.info(
            f"Advanced analytics initialized (LangSmith: {'enabled' if self.langsmith_client else 'disabled'})"
        )

    def start_chat_session(
        self, session_token: str, user_code: str, company: str
    ) -> None:
        """
        Start a new chat session run in LangSmith.

        Args:
            session_token (str): Unique token identifying the chat session.
            user_code (str): Identifier for the user.
            company (str): Name of the company.

        This method initializes a new run in LangSmith tracking if the client is available
        and the session is not already active.
        """
        analytics_logger.info(
            f"Starting chat session for company: {company}, user: {user_code}"
        )

        if (
            self.langsmith_client
            and hasattr(self, "active_chat_runs")
            and session_token not in self.active_chat_runs
        ):
            try:
                import uuid
                from datetime import datetime

                run_id = str(uuid.uuid4())

                self.langsmith_client.create_run(
                    id=run_id,
                    name="ChatCV-Conversation",
                    run_type="chain",
                    inputs={"user": company, "user_code": user_code},
                    project_name=os.getenv("LANGSMITH_PROJECT", "ChatCV"),
                    extra={
                        "user_code": user_code,
                        "company": company,
                        "session_start": datetime.utcnow().isoformat(),
                        "session_type": "conversation",
                    },
                    tags=["chatcv", "conversation", company.lower().replace(" ", "-")],
                )

                if not hasattr(self, "active_chat_runs"):
                    self.active_chat_runs = {}

                self.active_chat_runs[session_token] = {
                    "run_id": run_id,
                    "query_count": 0,
                    "conversation": [],
                }
                analytics_logger.info(f"Conversation tracking started for {company}")

            except Exception as e:
                analytics_logger.error(
                    f"LangSmith session start error ({e.__class__.__name__}): {e!s}"
                )
        else:
            # Log session start even without LangSmith
            analytics_logger.info(
                f"Chat session started for {company} (LangSmith disabled)"
            )

            # Store basic session info
            if not hasattr(self, "active_chat_runs"):
                self.active_chat_runs = {}

            self.active_chat_runs[session_token] = {
                "run_id": None,
                "query_count": 0,
                "conversation": [],
                "company": company,
                "user_code": user_code,
            }

    def end_chat_session(self, session_token: str) -> None:
        """
        End an active chat session.

        Args:
            session_token (str): Unique token identifying the chat session.

        Removes the session from active tracking and logs the number of queries processed.
        """
        if hasattr(self, "active_chat_runs") and session_token in self.active_chat_runs:
            session_info = self.active_chat_runs[session_token]
            query_count = session_info.get("query_count", 0)
            company = session_info.get("company", "Unknown")

            analytics_logger.info(
                f"Ending chat session for {company}, {query_count} queries processed"
            )

            # End LangSmith run if available
            if self.langsmith_client and session_info.get("run_id"):
                try:
                    self.langsmith_client.update_run(
                        session_info["run_id"],
                        outputs={"total_queries": query_count},
                        end_time=datetime.utcnow(),
                    )
                except Exception as e:
                    analytics_logger.warning(f"Failed to end LangSmith run: {e}")

            del self.active_chat_runs[session_token]

    def log_chat_interaction_advanced(
        self,
        user_code: str,
        company: str,
        query: str,
        response: str,
        response_time: float,
        sources: list,
        metadata: dict,
        session_token: str,
    ) -> None:
        """
        Log a chat interaction using standard logging.

        Args:
            user_code (str): Identifier for the user.
            company (str): Name of the company.
            query (str): User's query text.
            response (str): Response text.
            response_time (float): Time taken to generate the response in seconds.
            sources (list): Source documents or references used.
            metadata (dict): Additional metadata.
            session_token (str): Unique token identifying the chat session.
        """
        analytics_logger.info(
            f"Chat interaction logged - Company: {company}, Response time: {response_time:.2f}s"
        )

        # Standard logging (existing)
        log_chat_interaction(user_code, company, query, response, response_time)

    def log_summary_request_advanced(
        self,
        user_code: str,
        company: str,
        style: str,
        summary_text: str,
        metadata: dict,
    ) -> None:
        """
        Log a summary request using standard logging.

        Args:
            user_code (str): Identifier for the user.
            company (str): Name of the company.
            style (str): Style of the summary requested.
            summary_text (str): Generated summary text.
            metadata (dict): Additional metadata.
        """
        summary_length = len(summary_text) if summary_text else 0
        analytics_logger.info(
            f"Summary request logged - Company: {company}, Style: {style}, Length: {summary_length}"
        )
        log_summary_request(user_code, company, style, summary_length)

    def log_job_matching_advanced(
        self,
        user_code: str,
        company: str,
        job_source: str,
        analysis_text: str,
        response_time: float,
        session_token: str,
        metadata: dict,
    ) -> None:
        """
        Log job matching analysis with advanced tracking.

        Args:
            user_code (str): Identifier for the user.
            company (str): Name of the company.
            job_source (str): Source of the job data.
            analysis_text (str): Text of the job analysis.
            response_time (float): Time taken for the analysis in seconds.
            session_token (str): Unique token identifying the chat session.
            metadata (dict): Additional metadata.
        """
        analytics_logger.info(
            f"Job matching analysis logged - Company: {company}, "
            f"Source: {job_source}, "
            f"Response time: {response_time:.2f}s"
        )

        try:
            # Prepare log data dictionary
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "user_code": user_code,
                "company": company,
                "job_source": job_source,
                "response_time": response_time,
                "analysis_length": len(analysis_text),
                "session_token": session_token,
                "metadata": metadata or {},
            }

            # Append log entry to analytics file
            with open(self.analytics_file, "a") as f:
                f.write(f"JOB_MATCH,{json.dumps(log_data)}\n")

        except Exception as e:
            analytics_logger.error(
                f"Error logging job matching analysis ({e.__class__.__name__}): {e!s}"
            )


def log_chat_interaction(
    user_code: str, company: str, query: str, response: str, response_time: float
) -> None:
    """
    Log a chat interaction for analytics.

    Args:
        user_code (str): Identifier for the user.
        company (str): Name of the company.
        query (str): User's query text.
        response (str): Response text.
        response_time (float): Time taken to generate the response in seconds.
    """
    try:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "chat",
            "user_code": user_code,
            "company": company,
            "query": query[:200],  # Limit query length for privacy
            "response": response[:500],  # Limit response length
            "response_time": response_time,
            "query_length": len(query),
            "response_length": len(response),
        }

        if config.logging.analytics_enabled:
            analytics_logger.info(f"ANALYTICS: {json.dumps(log_entry)}")

            # Also write to file
            with open(config.data.analytics_log_path, "a") as f:
                f.write(f"CHAT,{json.dumps(log_entry)}\n")

    except Exception as e:
        analytics_logger.error(
            f"Analytics logging error ({e.__class__.__name__}): {e!s}"
        )


def log_summary_request(
    user_code: str, company: str, style: str, summary_length: int
) -> None:
    """
    Log a summary request for analytics.

    Args:
        user_code (str): Identifier for the user.
        company (str): Name of the company.
        style (str): Style of the summary requested.
        summary_length (int): Length of the summary text.
    """
    try:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "summary",
            "user_code": user_code,
            "company": company,
            "style": style,
            "summary_length": summary_length,
        }

        if config.logging.analytics_enabled:
            analytics_logger.info(f"ANALYTICS: {json.dumps(log_entry)}")

            # Also write to file
            with open(config.data.analytics_log_path, "a") as f:
                f.write(f"SUMMARY,{json.dumps(log_entry)}\n")

    except Exception as e:
        analytics_logger.error(
            f"Analytics logging error ({e.__class__.__name__}): {e!s}"
        )


def log_login_event(user_code: str, company: str, success: bool) -> None:
    """
    Log login events for analytics.

    Args:
        user_code (str): Identifier for the user.
        company (str): Name of the company.
        success (bool): Whether the login was successful.
    """
    try:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "login",
            "user_code": user_code,
            "company": company,
            "success": success,
        }

        if config.logging.analytics_enabled:
            analytics_logger.info(f"ANALYTICS: {json.dumps(log_entry)}")

            # Also write to file
            with open(config.data.analytics_log_path, "a") as f:
                f.write(f"LOGIN,{json.dumps(log_entry)}\n")

    except Exception as e:
        analytics_logger.error(
            f"Analytics logging error ({e.__class__.__name__}): {e!s}"
        )


def get_analytics_summary() -> dict[str, Any]:
    """
    Generate a basic analytics summary from the log file.

    Returns:
        Dict[str, Any]: Summary data including total interactions, counts by type,
                        unique companies, and a list of companies.
                        Returns an error dict if data is unavailable or on failure.
    """
    try:
        if not os.path.exists(config.data.analytics_log_path):
            analytics_logger.warning(
                "Analytics log file not found for summary generation."
            )
            return {"error": "No analytics data found"}

        with open(config.data.analytics_log_path, "r") as f:
            lines = f.readlines()

        total_interactions = 0
        chat_count = 0
        summary_count = 0
        companies = set()

        for line in lines:
            try:
                if "," in line:
                    event_type, log_json = line.split(",", 1)
                    log_data = json.loads(log_json)
                    total_interactions += 1
                    companies.add(log_data.get("company", "Unknown"))

                    if event_type == "CHAT":
                        chat_count += 1
                    elif event_type == "SUMMARY":
                        summary_count += 1
            except Exception as e:
                # Skip malformed lines silently
                analytics_logger.warning(f"Malformed analytics line skipped: {e}")
                continue

        summary = {
            "total_interactions": total_interactions,
            "chat_interactions": chat_count,
            "summary_requests": summary_count,
            "unique_companies": len(companies),
            "companies": list(companies),
        }
        analytics_logger.info("Analytics summary generated successfully.")
        return summary
    except Exception as e:
        analytics_logger.error(
            f"Error generating analytics summary ({e.__class__.__name__}): {e!s}"
        )
        return {"error": str(e)}
