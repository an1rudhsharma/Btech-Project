"""Comprehensive tests for /api/status endpoint."""

from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def api_client():
    with TestClient(app) as client:
        yield client


@pytest.fixture
def patched_engine():
    mock_engine = MagicMock()
    mock_engine.get_model_status.return_value = {
        "churn": {"trained": False, "metrics": None},
        "marketing": {"trained": False, "metrics": None},
        "pricing": {"trained": False, "metrics": None},
        "sentiment": {"trained": False, "metrics": None},
    }
    with patch("app.engine.simulation._engine_instance", mock_engine), \
         patch("app.engine.simulation.get_simulation_engine", return_value=mock_engine):
        yield mock_engine


# ---------------------------------------------------------------------------
# GET /api/status - Application Status
# ---------------------------------------------------------------------------

class TestStatus:
    """Tests for the application status endpoint."""

    def test_status_endpoint_returns_200(self, api_client, patched_engine):
        resp = api_client.get("/api/status")
        assert resp.status_code == 200

    def test_status_contains_models_field(self, api_client, patched_engine):
        patched_engine.get_model_status.return_value = {
            "churn": {"trained": True, "metrics": {"accuracy": 0.85}},
            "marketing": {"trained": False, "metrics": None},
            "pricing": {"trained": False, "metrics": None},
            "sentiment": {"trained": True, "metrics": {"accuracy": 0.90}},
        }
        resp = api_client.get("/api/status")
        data = resp.json()
        assert "models" in data
        assert "app" in data

    def test_status_models_structure(self, api_client, patched_engine):
        patched_engine.get_model_status.return_value = {
            "churn": {"trained": True, "metrics": {"accuracy": 0.85}},
            "marketing": {"trained": True, "metrics": {"r2": 0.72}},
            "pricing": {"trained": True, "metrics": {"r2": 0.68}},
            "sentiment": {"trained": True, "metrics": {"accuracy": 0.90}},
        }
        resp = api_client.get("/api/status")
        data = resp.json()
        models = data["models"]
        for model_name in ["churn", "marketing", "pricing", "sentiment"]:
            assert model_name in models
            assert "trained" in models[model_name]
            assert "metrics" in models[model_name]

    def test_status_all_untrained(self, api_client, patched_engine):
        resp = api_client.get("/api/status")
        data = resp.json()
        for model_info in data["models"].values():
            assert model_info["trained"] is False

    def test_status_all_trained(self, api_client, patched_engine):
        patched_engine.get_model_status.return_value = {
            "churn": {"trained": True, "metrics": {"accuracy": 0.85}},
            "marketing": {"trained": True, "metrics": {"r2": 0.72}},
            "pricing": {"trained": True, "metrics": {"r2": 0.68}},
            "sentiment": {"trained": True, "metrics": {"accuracy": 0.90}},
        }
        resp = api_client.get("/api/status")
        data = resp.json()
        for model_info in data["models"].values():
            assert model_info["trained"] is True

    def test_status_returns_app_name(self, api_client, patched_engine):
        resp = api_client.get("/api/status")
        data = resp.json()
        assert "app" in data
        assert isinstance(data["app"], str)

    def test_status_get_only(self, api_client):
        """POST should not work on the status endpoint."""
        resp = api_client.post("/api/status")
        assert resp.status_code == 405

    def test_status_no_body_needed(self, api_client, patched_engine):
        resp = api_client.get("/api/status")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# CORS & Middleware Tests
# ---------------------------------------------------------------------------

class TestCORS:
    """Tests for CORS middleware configuration."""

    def test_cors_allows_configured_origin(self, api_client, patched_engine):
        resp = api_client.get("/api/status", headers={"Origin": "http://localhost:5173"})
        assert resp.status_code == 200

    def test_options_preflight(self, api_client):
        resp = api_client.options("/api/status", headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        })
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 404 Tests
# ---------------------------------------------------------------------------

class TestNotFound:
    """Tests for non-existent endpoints."""

    def test_unknown_endpoint(self, api_client):
        resp = api_client.get("/api/nonexistent")
        assert resp.status_code == 404

    def test_unknown_nested_endpoint(self, api_client):
        resp = api_client.get("/api/simulate/unknown/deep/path")
        assert resp.status_code in (404, 405)

    def test_root_path(self, api_client):
        resp = api_client.get("/")
        assert resp.status_code in (200, 404)
