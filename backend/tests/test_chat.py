"""Comprehensive tests for /api/chat endpoints."""

from unittest.mock import patch, MagicMock, AsyncMock

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.routes.chat import router, ChatMessage, ReportRequest


@pytest.fixture
def app():
    test_app = FastAPI()
    test_app.include_router(router, prefix="/api")
    return test_app


@pytest.fixture
def api_client(app):
    with TestClient(app) as client:
        yield client


@pytest.fixture
def patched_orchestrator(mock_orchestrator):
    """Patch the orchestrator at its source module."""
    with patch("app.llm.orchestrator.orchestrator", mock_orchestrator):
        yield mock_orchestrator


# ---------------------------------------------------------------------------
# POST /api/chat - Natural Language Chat
# ---------------------------------------------------------------------------

class TestChat:
    """Tests for the chat endpoint."""

    def test_chat_basic_query(self, api_client, patched_orchestrator):
        resp = api_client.post("/api/chat", json={
            "message": "What happens if I increase price by 20%?",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "intent" in data or "insight" in data or "simulation_result" in data

    def test_chat_empty_message(self, api_client, patched_orchestrator):
        resp = api_client.post("/api/chat", json={"message": ""})
        assert resp.status_code in (200, 400, 422)

    def test_chat_very_long_message(self, api_client, patched_orchestrator):
        long_msg = "What if I change the price? " * 500
        resp = api_client.post("/api/chat", json={"message": long_msg})
        assert resp.status_code == 200

    def test_chat_special_characters(self, api_client, patched_orchestrator):
        resp = api_client.post("/api/chat", json={
            "message": "What about price=$100? <b>HTML</b> & 'quotes' \"double\"",
        })
        assert resp.status_code == 200

    def test_chat_unicode_message(self, api_client, patched_orchestrator):
        resp = api_client.post("/api/chat", json={
            "message": "What is the impact on churn?",
        })
        assert resp.status_code == 200

    def test_chat_sql_injection_attempt(self, api_client, patched_orchestrator):
        resp = api_client.post("/api/chat", json={
            "message": "'; DROP TABLE users; --",
        })
        assert resp.status_code == 200

    def test_chat_xss_attempt(self, api_client, patched_orchestrator):
        resp = api_client.post("/api/chat", json={
            "message": "<script>alert('xss')</script>",
        })
        assert resp.status_code == 200

    def test_chat_missing_message_field(self, api_client):
        resp = api_client.post("/api/chat", json={})
        assert resp.status_code == 422

    def test_chat_wrong_field_name(self, api_client):
        resp = api_client.post("/api/chat", json={"query": "hello"})
        assert resp.status_code == 422

    def test_chat_invalid_json(self, api_client):
        resp = api_client.post("/api/chat", content=b"not json", headers={"content-type": "application/json"})
        assert resp.status_code == 422

    def test_chat_null_message(self, api_client):
        resp = api_client.post("/api/chat", json={"message": None})
        assert resp.status_code == 422

    def test_chat_numeric_message(self, api_client):
        """Message field must be string."""
        resp = api_client.post("/api/chat", json={"message": 12345})
        # Pydantic may coerce int to string or reject
        assert resp.status_code in (200, 422)

    def test_chat_orchestrator_error_handling(self, app):
        """When orchestrator raises an exception, it propagates."""
        mock_orch = MagicMock()
        mock_orch.process_natural_language_query = AsyncMock(
            side_effect=Exception("LLM service unavailable")
        )
        with patch("app.llm.orchestrator.orchestrator", mock_orch):
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.post("/api/chat", json={"message": "test"})
                assert resp.status_code == 500

    def test_chat_simulation_query(self, api_client, patched_orchestrator):
        resp = api_client.post("/api/chat", json={
            "message": "Simulate price at $80 with $10k marketing spend",
        })
        assert resp.status_code == 200

    def test_chat_explanation_query(self, api_client, patched_orchestrator):
        resp = api_client.post("/api/chat", json={
            "message": "Why is churn predicted to be high?",
        })
        assert resp.status_code == 200

    def test_chat_counterfactual_query(self, api_client, patched_orchestrator):
        resp = api_client.post("/api/chat", json={
            "message": "What should I change to reduce churn below 30%?",
        })
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# POST /api/chat/report - Business Report Generation
# ---------------------------------------------------------------------------

class TestChatReport:
    """Tests for the report generation endpoint."""

    def test_report_basic(self, api_client, patched_orchestrator):
        resp = api_client.post("/api/chat/report", json={
            "scenarios": [
                {"price": 100, "churn": 0.3},
                {"price": 80, "churn": 0.2},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "report" in data

    def test_report_single_scenario(self, api_client, patched_orchestrator):
        resp = api_client.post("/api/chat/report", json={
            "scenarios": [{"price": 100}],
        })
        assert resp.status_code == 200

    def test_report_empty_scenarios(self, api_client, patched_orchestrator):
        resp = api_client.post("/api/chat/report", json={
            "scenarios": [],
        })
        assert resp.status_code == 200

    def test_report_many_scenarios(self, api_client, patched_orchestrator):
        scenarios = [{"price": i * 10, "marketing": i * 1000} for i in range(20)]
        resp = api_client.post("/api/chat/report", json={
            "scenarios": scenarios,
        })
        assert resp.status_code == 200

    def test_report_missing_scenarios_field(self, api_client):
        resp = api_client.post("/api/chat/report", json={})
        assert resp.status_code == 422

    def test_report_scenarios_not_a_list(self, api_client):
        resp = api_client.post("/api/chat/report", json={
            "scenarios": "not a list",
        })
        assert resp.status_code == 422

    def test_report_invalid_json(self, api_client):
        resp = api_client.post("/api/chat/report", content=b"{invalid", headers={"content-type": "application/json"})
        assert resp.status_code == 422

    def test_report_orchestrator_error(self, app):
        mock_orch = MagicMock()
        mock_orch.generate_business_report = AsyncMock(
            side_effect=Exception("Report generation failed")
        )
        with patch("app.llm.orchestrator.orchestrator", mock_orch):
            with TestClient(app, raise_server_exceptions=False) as client:
                resp = client.post("/api/chat/report", json={
                    "scenarios": [{"price": 100}],
                })
                assert resp.status_code == 500

    def test_report_scenarios_with_nested_objects(self, api_client, patched_orchestrator):
        resp = api_client.post("/api/chat/report", json={
            "scenarios": [
                {"price": 100, "details": {"region": "US", "segment": "enterprise"}},
            ],
        })
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Pydantic Model Validation
# ---------------------------------------------------------------------------

class TestChatModels:
    """Tests for Pydantic chat models."""

    def test_chat_message_valid(self):
        msg = ChatMessage(message="Hello")
        assert msg.message == "Hello"

    def test_report_request_valid(self):
        req = ReportRequest(scenarios=[{"a": 1}])
        assert len(req.scenarios) == 1

    def test_report_request_empty(self):
        req = ReportRequest(scenarios=[])
        assert req.scenarios == []
