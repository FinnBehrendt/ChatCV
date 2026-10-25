from unittest.mock import Mock

import pytest

from app.config import AppConfig
from app.modules.guardrails import QueryValidator


class TestQueryValidator:
    """Test input validation and guardrails."""

    @pytest.fixture
    def validator(self):
        """Create validator instance for testing."""
        config = AppConfig()
        logger = Mock()
        return QueryValidator(config, logger)

    def test_valid_query(self, validator):
        """Test validation of normal professional query."""
        query = "What is your experience with Python?"
        result = validator.validate_query_input(query)
        assert result == query

    def test_short_query_rejection(self, validator):
        """Test rejection of too short queries."""
        with pytest.raises(ValueError, match="at least 3 characters"):
            validator.validate_query_input("hi")

    def test_long_query_rejection(self, validator):
        """Test rejection of overly long queries."""
        long_query = "x" * 2000
        with pytest.raises(ValueError, match="too long"):
            validator.validate_query_input(long_query)

    def test_irrelevant_query_rejection(self, validator):
        """Test rejection of irrelevant queries."""
        with pytest.raises(ValueError, match="professional qualifications"):
            validator.validate_query_input("What's the weather like?")
