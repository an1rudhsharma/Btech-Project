"""Comprehensive tests for /api/counterfactual endpoints."""

from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.routes.counterfactual import router, CounterfactualRequest


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
def mock_cf_engine_available():
    """Mock counterfactual engine that has models available."""
    engine = MagicMock()
    engine.is_available.return_value = True
    engine.generate_counterfactuals.return_value = {
        "status": "success",
        "original": {"price": 120, "usage": 30, "num_features": 3},
        "counterfactuals": [
            {
                "action": "decrease price from 120 to 95; increase usage from 30 to 60",
                "changes": [
                    {"feature": "price", "from": 120, "to": 95, "direction": "decrease", "change_pct": -20.8},
                    {"feature": "usage", "from": 30, "to": 60, "direction": "increase", "change_pct": 100.0},
                ],
                "new_predicted_outcome": 0.15,
                "feasibility": 0.62,
            }
        ],
        "total_generated": 1,
    }
    return engine


@pytest.fixture
def mock_cf_engine_unavailable():
    """Mock counterfactual engine with no models available."""
    engine = MagicMock()
    engine.is_available.return_value = False
    return engine


# ---------------------------------------------------------------------------
# POST /api/counterfactual - Generate Counterfactuals
# ---------------------------------------------------------------------------

class TestCounterfactual:
    """Tests for the counterfactual generation endpoint."""

    def test_counterfactual_basic(self, api_client, mock_cf_engine_available):
        with patch("app.engine.counterfactual.counterfactual_engine", mock_cf_engine_available):
            resp = api_client.post("/api/counterfactual", json={
                "model_name": "churn",
                "scenario": {"price": 120, "usage": 30, "num_features": 3},
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "success"
            assert "counterfactuals" in data
            assert len(data["counterfactuals"]) > 0

    def test_counterfactual_model_not_available(self, api_client, mock_cf_engine_unavailable):
        with patch("app.engine.counterfactual.counterfactual_engine", mock_cf_engine_unavailable):
            resp = api_client.post("/api/counterfactual", json={
                "model_name": "churn",
                "scenario": {"price": 120, "usage": 30},
            })
            assert resp.status_code == 400
            assert "not available" in resp.json()["detail"].lower()

    def test_counterfactual_unknown_model(self, api_client, mock_cf_engine_unavailable):
        with patch("app.engine.counterfactual.counterfactual_engine", mock_cf_engine_unavailable):
            resp = api_client.post("/api/counterfactual", json={
                "model_name": "nonexistent",
                "scenario": {"price": 100},
            })
            assert resp.status_code == 400

    def test_counterfactual_with_custom_total_cfs(self, api_client, mock_cf_engine_available):
        with patch("app.engine.counterfactual.counterfactual_engine", mock_cf_engine_available):
            resp = api_client.post("/api/counterfactual", json={
                "model_name": "churn",
                "scenario": {"price": 120, "usage": 30},
                "total_cfs": 10,
            })
            assert resp.status_code == 200

    def test_counterfactual_with_desired_class(self, api_client, mock_cf_engine_available):
        with patch("app.engine.counterfactual.counterfactual_engine", mock_cf_engine_available):
            resp = api_client.post("/api/counterfactual", json={
                "model_name": "churn",
                "scenario": {"price": 120, "usage": 30},
                "desired_class": "opposite",
            })
            assert resp.status_code == 200

    def test_counterfactual_with_features_to_vary(self, api_client, mock_cf_engine_available):
        with patch("app.engine.counterfactual.counterfactual_engine", mock_cf_engine_available):
            resp = api_client.post("/api/counterfactual", json={
                "model_name": "churn",
                "scenario": {"price": 120, "usage": 30, "num_features": 5},
                "features_to_vary": ["price", "usage"],
            })
            assert resp.status_code == 200

    def test_counterfactual_with_permitted_range(self, api_client, mock_cf_engine_available):
        with patch("app.engine.counterfactual.counterfactual_engine", mock_cf_engine_available):
            resp = api_client.post("/api/counterfactual", json={
                "model_name": "churn",
                "scenario": {"price": 120, "usage": 30},
                "permitted_range": {"price": [50, 200], "usage": [0, 100]},
            })
            assert resp.status_code == 200

    def test_counterfactual_empty_scenario(self, api_client, mock_cf_engine_available):
        with patch("app.engine.counterfactual.counterfactual_engine", mock_cf_engine_available):
            resp = api_client.post("/api/counterfactual", json={
                "model_name": "churn",
                "scenario": {},
            })
            assert resp.status_code == 200

    def test_counterfactual_zero_total_cfs(self, api_client, mock_cf_engine_available):
        with patch("app.engine.counterfactual.counterfactual_engine", mock_cf_engine_available):
            resp = api_client.post("/api/counterfactual", json={
                "model_name": "churn",
                "scenario": {"price": 100},
                "total_cfs": 0,
            })
            assert resp.status_code == 200

    def test_counterfactual_negative_total_cfs(self, api_client, mock_cf_engine_available):
        with patch("app.engine.counterfactual.counterfactual_engine", mock_cf_engine_available):
            resp = api_client.post("/api/counterfactual", json={
                "model_name": "churn",
                "scenario": {"price": 100},
                "total_cfs": -1,
            })
            assert resp.status_code in (200, 400, 422)

    def test_counterfactual_very_large_total_cfs(self, api_client, mock_cf_engine_available):
        with patch("app.engine.counterfactual.counterfactual_engine", mock_cf_engine_available):
            resp = api_client.post("/api/counterfactual", json={
                "model_name": "churn",
                "scenario": {"price": 100},
                "total_cfs": 1000,
            })
            assert resp.status_code == 200

    def test_counterfactual_missing_scenario(self, api_client):
        resp = api_client.post("/api/counterfactual", json={
            "model_name": "churn",
        })
        assert resp.status_code == 422

    def test_counterfactual_invalid_json(self, api_client):
        resp = api_client.post("/api/counterfactual", content=b"not json", headers={"content-type": "application/json"})
        assert resp.status_code == 422

    def test_counterfactual_scenario_with_string_values(self, api_client, mock_cf_engine_available):
        """Scenario with non-numeric values."""
        with patch("app.engine.counterfactual.counterfactual_engine", mock_cf_engine_available):
            resp = api_client.post("/api/counterfactual", json={
                "model_name": "churn",
                "scenario": {"price": "expensive", "usage": "high"},
            })
            assert resp.status_code == 200

    def test_counterfactual_pricing_model(self, api_client, mock_cf_engine_available):
        with patch("app.engine.counterfactual.counterfactual_engine", mock_cf_engine_available):
            resp = api_client.post("/api/counterfactual", json={
                "model_name": "pricing",
                "scenario": {"price": 200, "num_features": 10},
                "desired_class": "opposite",
            })
            assert resp.status_code == 200

    def test_counterfactual_marketing_model(self, api_client, mock_cf_engine_available):
        with patch("app.engine.counterfactual.counterfactual_engine", mock_cf_engine_available):
            resp = api_client.post("/api/counterfactual", json={
                "model_name": "marketing",
                "scenario": {"marketing_spend": 5000, "impressions": 10000, "clicks": 500},
            })
            assert resp.status_code == 200

    def test_counterfactual_no_counterfactuals_found(self, api_client):
        engine = MagicMock()
        engine.is_available.return_value = True
        engine.generate_counterfactuals.return_value = {
            "status": "no_counterfactuals",
            "message": "Could not find valid counterfactual scenarios.",
        }
        with patch("app.engine.counterfactual.counterfactual_engine", engine):
            resp = api_client.post("/api/counterfactual", json={
                "model_name": "churn",
                "scenario": {"price": 100},
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "no_counterfactuals"

    def test_counterfactual_engine_error(self, api_client):
        engine = MagicMock()
        engine.is_available.return_value = True
        engine.generate_counterfactuals.return_value = {
            "status": "error",
            "message": "DiCE failed to generate counterfactuals",
        }
        with patch("app.engine.counterfactual.counterfactual_engine", engine):
            resp = api_client.post("/api/counterfactual", json={
                "model_name": "churn",
                "scenario": {"price": 100},
            })
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "error"

    def test_counterfactual_with_all_options(self, api_client, mock_cf_engine_available):
        with patch("app.engine.counterfactual.counterfactual_engine", mock_cf_engine_available):
            resp = api_client.post("/api/counterfactual", json={
                "model_name": "churn",
                "scenario": {"price": 150, "usage": 20, "num_features": 2, "marketing_spend": 3000},
                "total_cfs": 5,
                "desired_class": "opposite",
                "features_to_vary": ["price", "marketing_spend"],
                "permitted_range": {"price": [50, 300], "marketing_spend": [1000, 20000]},
            })
            assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Pydantic Model Validation
# ---------------------------------------------------------------------------

class TestCounterfactualRequestModel:
    """Tests for CounterfactualRequest Pydantic model."""

    def test_defaults(self):
        req = CounterfactualRequest(scenario={"price": 100})
        assert req.model_name == "churn"
        assert req.total_cfs == 4
        assert req.desired_class == "opposite"
        assert req.features_to_vary is None
        assert req.permitted_range is None

    def test_full_specification(self):
        req = CounterfactualRequest(
            model_name="pricing",
            scenario={"price": 200, "demand": 500},
            total_cfs=8,
            desired_class="opposite",
            features_to_vary=["price"],
            permitted_range={"price": [100, 300]},
        )
        assert req.model_name == "pricing"
        assert req.total_cfs == 8
        assert req.features_to_vary == ["price"]
