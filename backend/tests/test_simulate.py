"""Comprehensive tests for /api/simulate endpoints."""

from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.routes.simulate import router, ScenarioInput, ComparisonInput

SIMULATION_ENGINE_PATH = "app.engine.simulation.get_simulation_engine"


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
def patched_engine(mock_simulation_engine):
    """Patch the simulation engine used in lazy imports."""
    with patch("app.engine.simulation._engine_instance", mock_simulation_engine), \
         patch("app.engine.simulation.get_simulation_engine", return_value=mock_simulation_engine):
        yield mock_simulation_engine


# ---------------------------------------------------------------------------
# POST /api/simulate - Single Scenario Simulation
# ---------------------------------------------------------------------------

class TestSimulate:
    """Tests for the single simulation endpoint."""

    def test_simulate_with_defaults(self, api_client, patched_engine):
        resp = api_client.post("/api/simulate", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "pricing" in data or "churn" in data

    def test_simulate_with_all_parameters(self, api_client, patched_engine):
        resp = api_client.post("/api/simulate", json={
            "price": 150.0,
            "marketing_spend": 10000.0,
            "num_features": 8.0,
            "usage": 75.0,
            "impressions": 20000.0,
            "clicks": 1000.0,
            "text": "Great product, highly recommend!",
            "baseline_price": 100.0,
        })
        assert resp.status_code == 200

    def test_simulate_with_zero_values(self, api_client, patched_engine):
        resp = api_client.post("/api/simulate", json={
            "price": 0.0,
            "marketing_spend": 0.0,
            "num_features": 0.0,
            "usage": 0.0,
            "impressions": 0.0,
            "clicks": 0.0,
        })
        assert resp.status_code == 200

    def test_simulate_with_negative_values(self, api_client, patched_engine):
        resp = api_client.post("/api/simulate", json={
            "price": -50.0,
            "marketing_spend": -1000.0,
        })
        assert resp.status_code == 200

    def test_simulate_with_very_large_values(self, api_client, patched_engine):
        resp = api_client.post("/api/simulate", json={
            "price": 1e12,
            "marketing_spend": 1e15,
            "impressions": 1e18,
        })
        assert resp.status_code == 200

    def test_simulate_with_float_precision(self, api_client, patched_engine):
        resp = api_client.post("/api/simulate", json={
            "price": 99.99999999,
            "marketing_spend": 0.0000001,
        })
        assert resp.status_code == 200

    def test_simulate_with_empty_text(self, api_client, patched_engine):
        resp = api_client.post("/api/simulate", json={
            "text": "",
        })
        assert resp.status_code == 200

    def test_simulate_with_very_long_text(self, api_client, patched_engine):
        resp = api_client.post("/api/simulate", json={
            "text": "word " * 10000,
        })
        assert resp.status_code == 200

    def test_simulate_with_special_characters_in_text(self, api_client, patched_engine):
        resp = api_client.post("/api/simulate", json={
            "text": "Hello! @#$%^&*() <script>alert('xss')</script>",
        })
        assert resp.status_code == 200

    def test_simulate_with_unicode_text(self, api_client, patched_engine):
        resp = api_client.post("/api/simulate", json={
            "text": "Produit excellent! Sehr gut!",
        })
        assert resp.status_code == 200

    def test_simulate_invalid_json(self, api_client):
        resp = api_client.post("/api/simulate", content=b"not json", headers={"content-type": "application/json"})
        assert resp.status_code == 422

    def test_simulate_wrong_field_types(self, api_client):
        resp = api_client.post("/api/simulate", json={
            "price": "not a number",
        })
        assert resp.status_code == 422

    def test_simulate_extra_unknown_fields(self, api_client, patched_engine):
        """Extra fields should be ignored or rejected."""
        resp = api_client.post("/api/simulate", json={
            "price": 100.0,
            "unknown_field": "hello",
            "another_extra": 42,
        })
        assert resp.status_code == 200

    def test_simulate_returns_propagation_trace(self, api_client, patched_engine):
        resp = api_client.post("/api/simulate", json={"price": 100})
        data = resp.json()
        assert "propagation_trace" in data


# ---------------------------------------------------------------------------
# POST /api/simulate/compare - Scenario Comparison
# ---------------------------------------------------------------------------

class TestSimulateCompare:
    """Tests for the comparison simulation endpoint."""

    def test_compare_basic(self, api_client, patched_engine):
        resp = api_client.post("/api/simulate/compare", json={
            "baseline": {"price": 100.0, "marketing_spend": 5000.0},
            "scenario": {"price": 80.0, "marketing_spend": 8000.0},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "baseline" in data
        assert "scenario" in data
        assert "deltas" in data

    def test_compare_identical_scenarios(self, api_client, patched_engine):
        same = {"price": 100.0, "marketing_spend": 5000.0}
        resp = api_client.post("/api/simulate/compare", json={
            "baseline": same,
            "scenario": same,
        })
        assert resp.status_code == 200

    def test_compare_extreme_difference(self, api_client, patched_engine):
        resp = api_client.post("/api/simulate/compare", json={
            "baseline": {"price": 1.0, "marketing_spend": 0.0},
            "scenario": {"price": 10000.0, "marketing_spend": 1e9},
        })
        assert resp.status_code == 200

    def test_compare_with_text(self, api_client, patched_engine):
        resp = api_client.post("/api/simulate/compare", json={
            "baseline": {"price": 100.0},
            "scenario": {"price": 80.0},
            "text": "Customer feedback about price change",
        })
        assert resp.status_code == 200

    def test_compare_missing_scenario(self, api_client):
        resp = api_client.post("/api/simulate/compare", json={
            "baseline": {"price": 100.0},
        })
        assert resp.status_code == 422

    def test_compare_empty_body(self, api_client):
        resp = api_client.post("/api/simulate/compare", json={})
        assert resp.status_code == 422

    def test_compare_invalid_types(self, api_client):
        resp = api_client.post("/api/simulate/compare", json={
            "baseline": "not an object",
            "scenario": {"price": 100},
        })
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /api/simulate/batch - Batch Simulation
# ---------------------------------------------------------------------------

class TestSimulateBatch:
    """Tests for the batch simulation endpoint."""

    def test_batch_single_scenario(self, api_client, patched_engine):
        resp = api_client.post("/api/simulate/batch", json=[
            {"price": 100.0},
        ])
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 1
        assert len(data["scenarios"]) == 1

    def test_batch_multiple_scenarios(self, api_client, patched_engine):
        scenarios = [
            {"price": 80.0, "marketing_spend": 3000.0},
            {"price": 100.0, "marketing_spend": 5000.0},
            {"price": 120.0, "marketing_spend": 8000.0},
        ]
        resp = api_client.post("/api/simulate/batch", json=scenarios)
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 3

    def test_batch_empty_list(self, api_client, patched_engine):
        resp = api_client.post("/api/simulate/batch", json=[])
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0

    def test_batch_many_scenarios(self, api_client, patched_engine):
        """Stress test with many scenarios."""
        scenarios = [{"price": float(i * 10)} for i in range(50)]
        resp = api_client.post("/api/simulate/batch", json=scenarios)
        assert resp.status_code == 200
        assert resp.json()["count"] == 50

    def test_batch_with_varied_parameters(self, api_client, patched_engine):
        scenarios = [
            {"price": 50.0, "text": "Budget option"},
            {"price": 200.0, "marketing_spend": 20000.0, "text": "Premium"},
            {"price": 100.0, "usage": 90.0, "num_features": 15.0},
        ]
        resp = api_client.post("/api/simulate/batch", json=scenarios)
        assert resp.status_code == 200

    def test_batch_invalid_entry(self, api_client):
        resp = api_client.post("/api/simulate/batch", json=[
            {"price": "invalid"},
        ])
        assert resp.status_code == 422

    def test_batch_not_a_list(self, api_client):
        resp = api_client.post("/api/simulate/batch", json={"price": 100})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Pydantic Model Validation
# ---------------------------------------------------------------------------

class TestScenarioInputValidation:
    """Tests for ScenarioInput model validation."""

    def test_scenario_defaults(self):
        s = ScenarioInput()
        assert s.price == 100.0
        assert s.marketing_spend == 5000.0
        assert s.num_features == 5.0
        assert s.usage == 50.0
        assert s.impressions == 10000.0
        assert s.clicks == 500.0
        assert s.text == "Product is good"
        assert s.baseline_price is None

    def test_scenario_with_all_fields(self):
        s = ScenarioInput(
            price=200.0,
            marketing_spend=15000.0,
            num_features=10.0,
            usage=80.0,
            impressions=50000.0,
            clicks=2000.0,
            text="Excellent",
            baseline_price=150.0,
        )
        assert s.price == 200.0
        assert s.baseline_price == 150.0

    def test_comparison_input_requires_scenario(self):
        with pytest.raises(Exception):
            ComparisonInput()  # Missing required 'scenario'
