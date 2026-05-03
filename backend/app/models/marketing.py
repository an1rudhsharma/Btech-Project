"""Marketing Impact Model - LightGBM Regressor for CTR/Conversion/ROI."""

import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.models.base import BaseModel


class MarketingModel(BaseModel):
    """Predicts marketing metrics (conversion rate, CTR, ROI).

    Outputs predicted_conversion that feeds into Churn model.
    Outputs marketing_effect that feeds into Sentiment model.
    """

    FEATURE_ROLES = ["marketing_spend", "impressions", "clicks", "price"]
    PROPAGATED_FEATURES = ["sentiment_score"]

    def __init__(self, model_dir: Path):
        super().__init__("marketing", model_dir)

    def get_all_features(self) -> list[str]:
        return self.FEATURE_ROLES + self.PROPAGATED_FEATURES

    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        self.feature_names = list(X.columns)

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        y_pred = self.model.predict(X_val)

        cv_scores = cross_val_score(
            lgb.LGBMRegressor(n_estimators=200, verbose=-1),
            X, y, cv=5, scoring="r2"
        )

        self.training_metrics = {
            "r2_score": float(r2_score(y_val, y_pred)),
            "mae": float(mean_absolute_error(y_val, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_val, y_pred))),
            "cv_r2_mean": float(cv_scores.mean()),
            "cv_r2_std": float(cv_scores.std()),
            "n_samples": len(X),
            "n_features": len(self.feature_names),
        }

        self.is_trained = True
        self.save()
        return self.training_metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Marketing model not trained")
        X_aligned = self._align_features(X)
        return self.model.predict(X_aligned)

    def predict_with_engagement(self, X: pd.DataFrame) -> dict:
        """Predict conversion and also output engagement signal for sentiment."""
        conversion = self.predict(X)
        engagement = np.clip(conversion * 2.0, 0, 1)  # normalized engagement
        return {
            "conversion": conversion,
            "engagement": engagement,
        }

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=X.index)
        for col in self.feature_names:
            if col in X.columns:
                result[col] = X[col]
            else:
                result[col] = 0.0
        return result
