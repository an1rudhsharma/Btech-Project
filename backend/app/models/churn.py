"""Churn Prediction Model - LightGBM Classifier."""

import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from app.models.base import BaseModel


class ChurnModel(BaseModel):
    """Predicts customer churn probability using LightGBM.

    Accepts propagated signals from other models:
    - sentiment_score (from Sentiment model)
    - predicted_demand (from Pricing model)
    - predicted_conversion (from Marketing model)
    """

    FEATURE_ROLES = [
        "price", "marketing_spend", "num_features", "usage", "tenure", "satisfaction"
    ]
    PROPAGATED_FEATURES = ["sentiment_score", "predicted_demand", "predicted_conversion"]

    def __init__(self, model_dir: Path):
        super().__init__("churn", model_dir)
        self.label_encoder = LabelEncoder()

    def get_all_features(self) -> list[str]:
        return self.FEATURE_ROLES + self.PROPAGATED_FEATURES

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        if y.dtype == object:
            y = pd.Series(self.label_encoder.fit_transform(y))

        self.feature_names = list(X.columns)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model = lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=7,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        y_pred = self.model.predict(X_val)
        y_proba = self.model.predict_proba(X_val)[:, 1]

        cv_scores = cross_val_score(
            lgb.LGBMClassifier(n_estimators=200, verbose=-1, class_weight="balanced"),
            X, y, cv=5, scoring="roc_auc"
        )

        self.training_metrics = {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "f1_score": float(f1_score(y_val, y_pred, average="weighted")),
            "roc_auc": float(roc_auc_score(y_val, y_proba)),
            "cv_auc_mean": float(cv_scores.mean()),
            "cv_auc_std": float(cv_scores.std()),
            "n_samples": len(X),
            "n_features": len(self.feature_names),
        }

        self.is_trained = True
        self.save()
        return self.training_metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Churn model not trained")
        X_aligned = self._align_features(X)
        return self.model.predict(X_aligned)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Churn model not trained")
        X_aligned = self._align_features(X)
        return self.model.predict_proba(X_aligned)

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure input has all expected features in correct order."""
        result = pd.DataFrame(index=X.index)
        for col in self.feature_names:
            if col in X.columns:
                result[col] = X[col]
            else:
                result[col] = 0.0
        return result
