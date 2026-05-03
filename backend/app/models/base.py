"""Base model class with SHAP integration and persistence."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional
import joblib
import numpy as np
import pandas as pd
import shap


class BaseModel(ABC):
    """Abstract base for all simulation models."""

    def __init__(self, name: str, model_dir: Path):
        self.name = name
        self.model_dir = model_dir
        self.model = None
        self.is_trained = False
        self.feature_names: list[str] = []
        self.training_metrics: dict = {}
        self._explainer = None

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Train the model. Returns metrics dict."""
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        ...

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Predict probabilities (classification models only)."""
        return None

    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Get SHAP values for explainability."""
        if self._explainer is None and self.model is not None:
            self._explainer = shap.TreeExplainer(self.model)
        if self._explainer is None:
            return np.zeros((len(X), len(self.feature_names)))
        return self._explainer.shap_values(X)

    def get_shap_summary(self, X: pd.DataFrame) -> list[dict[str, Any]]:
        """Get top feature contributions as a readable list."""
        shap_vals = self.get_shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # positive class for binary

        if len(shap_vals.shape) == 1:
            shap_vals = shap_vals.reshape(1, -1)

        mean_abs = np.abs(shap_vals).mean(axis=0)
        sorted_idx = np.argsort(mean_abs)[::-1]

        summary = []
        for idx in sorted_idx[:5]:
            summary.append({
                "feature": self.feature_names[idx],
                "importance": float(mean_abs[idx]),
                "direction": "positive" if shap_vals[0, idx] > 0 else "negative",
            })
        return summary

    def save(self):
        """Persist model to disk."""
        if self.model is None:
            return
        path = self.model_dir / f"{self.name}.joblib"
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
            "metrics": self.training_metrics,
        }, path)

    def load(self) -> bool:
        """Load model from disk. Returns True if successful."""
        path = self.model_dir / f"{self.name}.joblib"
        if not path.exists():
            return False
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.training_metrics = data["metrics"]
        self.is_trained = True
        self._explainer = None
        return True
