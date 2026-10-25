from fastapi.testclient import TestClient

from app.config import config
from main import app

client = TestClient(app)


class TestBasicRoutes:
    """Test basic application routes and health checks."""

    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "ChatCV-OSS"

    def test_main_page_loads(self):
        """Test that main page loads successfully."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


class TestAuthentication:
    """Test authentication endpoints."""

    def test_login_invalid_code(self):
        """Test login with invalid invite code."""
        response = client.post("/auth/login", json={"invite_code": "invalid"})
        assert response.status_code == 401

    def test_logout_endpoint(self):
        """Test logout endpoint."""
        response = client.post("/auth/logout")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestConfiguration:
    """Test configuration management."""

    def test_config_loading(self):
        """Test that configuration loads properly."""
        assert config.llm.model is not None
        assert config.candidate.name is not None

    def test_environment_variables(self):
        """Test environment variable handling."""
        assert hasattr(config, "environment")
        assert config.environment in ["development", "production"]
