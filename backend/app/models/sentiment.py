"""Sentiment Analysis Model - DistilBERT with fallback."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from app.models.base import BaseModel


class SentimentModel(BaseModel):
    """Analyzes customer sentiment using DistilBERT.

    Outputs a sentiment_score (0-1) that feeds into the Churn model.
    Can also accept price_change_pct and marketing_intensity as contextual signals.
    """

    def __init__(self, model_dir: Path):
        super().__init__("sentiment", model_dir)
        self._pipeline = None
        self._use_transformer = True

    def _load_pipeline(self):
        """Lazy-load the transformer pipeline."""
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
            self._pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                truncation=True,
                max_length=512,
            )
            self._use_transformer = True
        except Exception:
            self._use_transformer = False

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """For DistilBERT, 'training' means loading the pretrained model.

        If labeled data is provided, we evaluate on it to report metrics.
        """
        self._load_pipeline()

        if self._use_transformer and len(X) > 0:
            text_col = X.columns[0] if len(X.columns) > 0 else None
            if text_col:
                sample = X[text_col].head(100).tolist()
                preds = self._batch_predict_text(sample)

                if y is not None and len(y) > 0:
                    y_binary = self._to_binary(y)
                    pred_binary = (np.array(preds[:len(y_binary)]) > 0.5).astype(int)
                    accuracy = float((pred_binary == y_binary[:len(pred_binary)]).mean())
                else:
                    accuracy = None

                self.training_metrics = {
                    "model_type": "distilbert-base-uncased-finetuned-sst-2-english",
                    "accuracy_on_sample": accuracy,
                    "n_samples_evaluated": min(100, len(X)),
                    "status": "pretrained_loaded",
                }
        else:
            self.training_metrics = {
                "model_type": "fallback_rule_based",
                "status": "no_transformer_available",
            }

        self.is_trained = True
        self.feature_names = ["text"]
        return self.training_metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict sentiment scores (0-1) for text inputs."""
        self._load_pipeline()
        if isinstance(X, pd.DataFrame):
            texts = X.iloc[:, 0].tolist() if len(X.columns) > 0 else []
        else:
            texts = list(X)
        return np.array(self._batch_predict_text(texts))

    def predict_single(self, text: str, context: Optional[dict] = None) -> float:
        """Predict sentiment for a single text with optional business context."""
        self._load_pipeline()
        base_score = self._predict_text(text)

        if context:
            price_change = context.get("price_change_pct", 0.0)
            marketing_intensity = context.get("marketing_intensity", 1.0)
            base_score = self._adjust_for_context(base_score, price_change, marketing_intensity)

        return float(np.clip(base_score, 0.0, 1.0))

    def _predict_text(self, text: str) -> float:
        """Get sentiment score for single text."""
        if self._use_transformer and self._pipeline:
            result = self._pipeline(text[:512])[0]
            if result["label"] == "POSITIVE":
                return result["score"]
            return 1.0 - result["score"]
        return self._rule_based_sentiment(text)

    def _batch_predict_text(self, texts: list[str]) -> list[float]:
        """Batch predict sentiment scores."""
        if not texts:
            return []
        if self._use_transformer and self._pipeline:
            truncated = [t[:512] for t in texts]
            results = self._pipeline(truncated, batch_size=32)
            scores = []
            for r in results:
                if r["label"] == "POSITIVE":
                    scores.append(r["score"])
                else:
                    scores.append(1.0 - r["score"])
            return scores
        return [self._rule_based_sentiment(t) for t in texts]

    def _adjust_for_context(
        self, base_score: float, price_change_pct: float, marketing_intensity: float
    ) -> float:
        """Adjust sentiment based on business context signals.
        Learned heuristic: price increases dampen sentiment, marketing boosts it slightly.
        """
        adjustment = 0.0
        adjustment -= price_change_pct * 0.002
        adjustment += (marketing_intensity - 1.0) * 0.05
        return base_score + adjustment

    @staticmethod
    def _rule_based_sentiment(text: str) -> float:
        """Simple fallback when no transformer is available."""
        positive_words = {"good", "great", "excellent", "love", "best", "amazing", "happy", "satisfied"}
        negative_words = {"bad", "terrible", "worst", "hate", "poor", "awful", "disappointed", "angry"}
        words = set(text.lower().split())
        pos = len(words & positive_words)
        neg = len(words & negative_words)
        total = pos + neg
        if total == 0:
            return 0.5
        return pos / total

    @staticmethod
    def _to_binary(y: pd.Series) -> np.ndarray:
        """Convert sentiment labels to binary (0/1)."""
        y_str = y.astype(str).str.lower()
        positive_labels = {"positive", "pos", "1", "good", "5", "4"}
        return np.array([1 if v in positive_labels else 0 for v in y_str])

    def get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Sentiment model uses text - return placeholder for compatibility."""
        return np.zeros((len(X), 1))

    def get_shap_summary(self, X: pd.DataFrame) -> list[dict]:
        return [{"feature": "text_content", "importance": 1.0, "direction": "varies"}]

    def save(self):
        """Sentiment model is pretrained - just save metadata."""
        import joblib
        path = self.model_dir / f"{self.name}.joblib"
        joblib.dump({
            "model": "distilbert",
            "feature_names": self.feature_names,
            "metrics": self.training_metrics,
        }, path)

    def load(self) -> bool:
        import joblib
        path = self.model_dir / f"{self.name}.joblib"
        if not path.exists():
            return False
        data = joblib.load(path)
        self.feature_names = data["feature_names"]
        self.training_metrics = data["metrics"]
        self.is_trained = True
        return True
