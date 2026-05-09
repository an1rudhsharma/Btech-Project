"""Shared fixtures for API test suite."""

import os
import io
import zipfile
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient


@pytest.fixture(scope="session", autouse=True)
def setup_env():
    """Set required environment variables before app import."""
    os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")
    yield


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary directory for test data files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "uploads").mkdir()
    return data_dir


@pytest.fixture
def client(tmp_data_dir):
    """FastAPI test client with mocked settings."""
    with patch("app.config.settings") as mock_settings:
        mock_settings.app_name = "Test App"
        mock_settings.groq_api_key = "test-key"
        mock_settings.model_dir = tmp_data_dir / "models"
        mock_settings.data_dir = tmp_data_dir
        mock_settings.cors_origins = ["http://localhost:3000"]
        mock_settings.max_upload_size_mb = 50
        mock_settings.llm_model = "llama-3.1-70b-versatile"
        (tmp_data_dir / "models").mkdir(exist_ok=True)

        from app.main import app
        with TestClient(app) as c:
            yield c


@pytest.fixture
def sample_churn_csv(tmp_path) -> Path:
    """Generate a valid churn dataset CSV."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "churn": np.random.randint(0, 2, n),
        "price": np.random.uniform(50, 200, n),
        "usage": np.random.uniform(1, 100, n),
        "num_features": np.random.randint(1, 20, n),
        "marketing_spend": np.random.uniform(100, 10000, n),
    })
    path = tmp_path / "churn_data.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_marketing_csv(tmp_path) -> Path:
    """Generate a valid marketing dataset CSV."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "conversion_rate": np.random.uniform(0, 0.3, n),
        "marketing_spend": np.random.uniform(1000, 50000, n),
        "impressions": np.random.randint(1000, 100000, n),
        "clicks": np.random.randint(10, 5000, n),
    })
    path = tmp_path / "marketing_data.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_pricing_csv(tmp_path) -> Path:
    """Generate a valid pricing dataset CSV."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "price": np.random.uniform(10, 500, n),
        "demand": np.random.uniform(50, 1000, n),
        "num_features": np.random.randint(1, 20, n),
    })
    path = tmp_path / "pricing_data.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def sample_sentiment_csv(tmp_path) -> Path:
    """Generate a valid sentiment dataset CSV."""
    texts = [
        "Great product, love it!",
        "Terrible service, never again",
        "Okay experience overall",
        "Amazing quality for the price",
        "Worst purchase I ever made",
    ] * 20
    labels = [1, 0, 1, 1, 0] * 20
    df = pd.DataFrame({"text": texts, "sentiment": labels})
    path = tmp_path / "sentiment_data.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def empty_csv(tmp_path) -> Path:
    """CSV file with headers but no rows."""
    path = tmp_path / "empty.csv"
    path.write_text("col1,col2,col3\n")
    return path


@pytest.fixture
def malformed_csv(tmp_path) -> Path:
    """CSV with inconsistent columns and garbage data."""
    path = tmp_path / "malformed.csv"
    path.write_text("a,b,c\n1,2\n3,4,5,6\nfoo,,\n")
    return path


@pytest.fixture
def binary_file(tmp_path) -> Path:
    """A binary (non-text) file masquerading as CSV."""
    path = tmp_path / "binary.csv"
    path.write_bytes(b"\x00\x01\x02\x03\xff\xfe\xfd" * 100)
    return path


@pytest.fixture
def large_csv(tmp_path) -> Path:
    """CSV that exceeds the upload size limit."""
    path = tmp_path / "large.csv"
    with open(path, "w") as f:
        f.write(",".join([f"col_{i}" for i in range(100)]) + "\n")
        row = ",".join(["x" * 1000 for _ in range(100)])
        for _ in range(600):
            f.write(row + "\n")
    return path


@pytest.fixture
def valid_zip(tmp_path, sample_churn_csv) -> Path:
    """ZIP containing a valid CSV."""
    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(sample_churn_csv, "churn_data.csv")
    return zip_path


@pytest.fixture
def empty_zip(tmp_path) -> Path:
    """ZIP containing no data files."""
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("readme.txt", "no data here")
    return zip_path


@pytest.fixture
def corrupt_zip(tmp_path) -> Path:
    """Invalid ZIP file (just random bytes)."""
    path = tmp_path / "corrupt.zip"
    path.write_bytes(b"PK\x03\x04" + b"\x00" * 50 + b"garbage")
    return path


@pytest.fixture
def mock_simulation_engine():
    """Mock the simulation engine for tests that don't need real ML."""
    engine = MagicMock()
    engine.models = {
        "churn": MagicMock(is_trained=True, training_metrics={"accuracy": 0.85}),
        "marketing": MagicMock(is_trained=True, training_metrics={"r2": 0.72}),
        "pricing": MagicMock(is_trained=False, training_metrics=None),
        "sentiment": MagicMock(is_trained=True, training_metrics={"accuracy": 0.90}),
    }
    engine.get_model_status.return_value = {
        "churn": {"trained": True, "metrics": {"accuracy": 0.85}},
        "marketing": {"trained": True, "metrics": {"r2": 0.72}},
        "pricing": {"trained": False, "metrics": None},
        "sentiment": {"trained": True, "metrics": {"accuracy": 0.90}},
    }
    engine.simulate.return_value = {
        "pricing": {"demand": 150.0, "revenue": 15000.0, "price_change_pct": -5.0, "elasticity": -1.2, "shap_drivers": []},
        "marketing": {"conversion_rate": 0.12, "engagement": 0.75, "shap_drivers": []},
        "sentiment": {"sentiment_score": 0.72, "label": "positive", "context_adjustments": {}},
        "churn": {"churn_probability": 0.35, "risk_level": "low", "shap_drivers": []},
        "propagation_trace": {},
        "scenario_input": {},
    }
    engine.simulate_comparison.return_value = {
        "baseline": {"churn": {"churn_probability": 0.5}},
        "scenario": {"churn": {"churn_probability": 0.3}},
        "deltas": {"churn": {"churn_probability": -0.2}},
    }
    return engine


@pytest.fixture
def mock_orchestrator():
    """Mock the LLM orchestrator for chat tests."""
    orch = MagicMock()
    orch.process_natural_language_query = AsyncMock(return_value={
        "intent": "simulate",
        "parameters": {"price": 100},
        "simulation_result": {"churn": {"churn_probability": 0.3}},
        "insight": "Lowering price reduces churn.",
    })
    orch.generate_business_report = AsyncMock(return_value="## Business Report\nAll is well.")
    return orch


def make_upload_file(path: Path, filename: str = None) -> tuple:
    """Helper to create file tuple for TestClient upload."""
    fname = filename or path.name
    return ("file", (fname, open(path, "rb"), "application/octet-stream"))


def make_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to CSV bytes."""
    return df.to_csv(index=False).encode("utf-8")
