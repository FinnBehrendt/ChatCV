import re
from logging import Logger

from app.config import AppConfig

# Optional Guardrails integration
try:
    from guardrails import Guard
    from guardrails.hub import ProfanityFree, ReadingTime, ToxicLanguage

    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False

    # Create mock classes when guardrails is not available
    class MockGuard:
        def use(self, *args, **kwargs):
            return self

        def validate(self, text):
            # Basic validation without guardrails
            if any(word in text.lower() for word in ["fuck", "shit", "damn"]):
                raise Exception("Content contains inappropriate language")
            return text

    class MockToxicLanguage:
        pass

    class MockProfanityFree:
        pass

    class MockReadingTime:
        pass

    Guard = MockGuard
    ToxicLanguage = MockToxicLanguage
    ProfanityFree = MockProfanityFree
    ReadingTime = MockReadingTime


class QueryValidator:
    """Validates and sanitizes input queries and generated responses."""

    def __init__(self, config: AppConfig, logger: Logger):
        """
        Initialize QueryValidator with configuration and logger.

        Args:
            config (object): Configuration object containing security settings.
            logger (object): Logger instance for logging messages.
        """
        self.config = config
        self.logger = logger

        if not GUARDRAILS_AVAILABLE:
            self.logger.warning(
                "Guardrails AI not available - using basic validation only"
            )

    def validate_query_input(self, query: str) -> str:
        """
        Validate and sanitize input queries using multiple guardrails.

        Args:
            query (str): Input query string.

        Returns:
            str: Validated and sanitized query string.

        Raises:
            ValueError: If the query fails any validation checks.
        """
        self.logger.debug(f"Validating query input, length: {len(query)}")

        try:
            # Apply content validation only if guardrails is available
            if GUARDRAILS_AVAILABLE:
                # Initialize validation guards
                toxic_guard = Guard().use(
                    ToxicLanguage, threshold=0.5, validation_method="sentence"
                )
                profanity_guard = Guard().use(ProfanityFree)

                # Apply content validation
                toxic_guard.validate(query)
                profanity_guard.validate(query)
            else:
                # Basic content validation without guardrails
                mock_guard = Guard()
                mock_guard.validate(query)

            # Length validation using config
            if len(query.strip()) < 3:
                raise ValueError("Query must be at least 3 characters long")

            if len(query) > self.config.security.max_query_length:
                raise ValueError(
                    f"Query too long. Maximum {self.config.security.max_query_length} characters allowed."
                )

            # Content relevance check
            irrelevant_keywords = ["weather", "sports", "cooking", "movies", "games"]
            if any(keyword in query.lower() for keyword in irrelevant_keywords):
                raise ValueError(
                    "Please ask questions related to professional qualifications and experience"
                )

            self.logger.debug("Query input validation succeeded")
            return query

        except Exception as e:
            error_message = str(e).lower()
            self.logger.warning(f"Query validation failed: {error_message}")
            if "injection" in error_message:
                raise ValueError(
                    "Invalid query format detected. Please rephrase your question"
                )
            elif any(term in error_message for term in ["profanity", "toxic:"]):
                raise ValueError("Please rephrase your question professionally")
            elif any(
                term in error_message
                for term in [
                    "query must be at least 3 characters long",
                    "query too long",
                    "please ask questions related to professional qualifications and experience",
                ]
            ):
                raise
            else:
                raise ValueError(
                    "Unable to process this query. Please rephrase your question"
                )

    def validate_response_output(self, response: str) -> str:
        """
        Validate generated responses using content guardrails.

        Args:
            response (str): Generated response text.

        Returns:
            str: Validated response string.
        """
        self.logger.debug(f"Validating response output, length: {len(response)}")

        try:
            if GUARDRAILS_AVAILABLE:
                # Initialize output validation guards
                toxic_guard = Guard().use(
                    ToxicLanguage, threshold=0.3, validation_method="sentence"
                )
                profanity_guard = Guard().use(ProfanityFree)
                reading_time_guard = Guard().use(ReadingTime, reading_time=0.5)

                toxic_guard.validate(response)
                profanity_guard.validate(response)
                reading_time_guard.validate(response)
            else:
                # Basic response validation without guardrails
                mock_guard = Guard()
                mock_guard.validate(response)
            return response
        except Exception as e:
            self.logger.warning(f"Response validation failed: {e!s}")
            return self.config.chat_fallback_response


class InputValidator:
    """Handles input validation for job descriptions"""

    def __init__(self, config: AppConfig, logger: Logger):
        """
        Initialize InputValidator with configuration and logger.

        Args:
            config (object): Configuration object containing security settings.
            logger (object): Logger instance for logging messages.
        """
        self.config = config
        self.logger = logger

    def validate_job_text(self, text: str) -> bool:
        """
        Validate the job description text input.

        Checks for length constraints, presence of key sections in English or German,
        and scans for potentially harmful HTML tags and event handlers.

        Args:
            text (str): Job description text to validate.

        Returns:
            bool: True if validation passes; False if input is empty or not a string.

        Raises:
            ValueError: If the text contains disallowed content or is too long,
                        or lacks required key sections.
        """
        if not text or not isinstance(text, str):
            return False

        text = text.strip()

        min_length = self.config.security.min_job_text_length
        max_length = self.config.security.max_job_text_length

        if len(text) < min_length:
            return False

        if len(text) > max_length:
            raise ValueError(
                f"Job description text is too long. Maximum {max_length} characters allowed."
            )
        # Check for presence of key sections in English or German
        english_keywords = (
            r"\b("
            r"job|position|role|responsibilities|skills|"
            r"qualifications|experience|requirements"
            r")\b"
        )
        german_keywords = (
            r"\b("
            r"arbeit|position|rolle|verantwortlichkeiten|fähigkeiten|"
            r"qualifikationen|erfahrung|anforderungen"
            r")\b"
        )
        if not (
            re.search(english_keywords, text, re.IGNORECASE)
            or re.search(german_keywords, text, re.IGNORECASE)
        ):
            raise ValueError(
                "Job description text must contain key sections like 'experience', 'position', 'role', etc."
            )

        # Extended XSS check for dangerous tags and event handlers
        dangerous_tags = [
            r"<script",
            r"<iframe",
            r"<object",
            r"<embed",
            r"<form",
            r"<img",
            r"<svg",
            r"<math",
            r"<link",
            r"<style",
            r"<video",
            r"<audio",
            r"<base",
            r"<meta",
            r"<body",
            r"<html",
            r"<input",
            r"<button",
        ]
        dangerous_events = [
            r"on\w+\s*=",  # e.g. onclick=, onload=, onerror= etc.
        ]

        for pattern in dangerous_tags:
            if re.search(pattern, text, re.IGNORECASE):
                raise ValueError(
                    "Job description text contains potentially harmful HTML tags."
                )

        for pattern in dangerous_events:
            if re.search(pattern, text, re.IGNORECASE):
                raise ValueError(
                    "Job description text contains potentially harmful event handlers."
                )

        self.logger.debug(
            f"Job description text validation passed, length: {len(text)}"
        )
        return True
