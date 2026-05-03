"""Pricing Impact Model - LightGBM Regressor for demand/revenue/elasticity."""

import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.models.base import BaseModel


class PricingModel(BaseModel):
    """Predicts demand and revenue based on pricing decisions.

    Outputs predicted_demand that feeds into Churn model.
    Outputs price_sentiment_impact that feeds into Sentiment model.
    """

    FEATURE_ROLES = ["price", "marketing_spend", "usage"]

    def __init__(self, model_dir: Path):
        super().__init__("pricing", model_dir)

    def get_all_features(self) -> list[str]:
        return self.FEATURE_ROLES

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
            raise RuntimeError("Pricing model not trained")
        X_aligned = self._align_features(X)
        return self.model.predict(X_aligned)

    def predict_with_impact(self, X: pd.DataFrame, baseline_price: float = 100.0) -> dict:
        """Predict demand and compute price impact signal for sentiment."""
        demand = self.predict(X)
        price = X["price"].values if "price" in X.columns else np.array([baseline_price])
        if baseline_price is None:
            baseline_price = 100.0
        price_change_pct = (price - baseline_price) / baseline_price * 100
        sentiment_impact = -price_change_pct * 0.01  # price up -> sentiment down
        return {
            "demand": demand,
            "revenue": demand * price,
            "price_change_pct": price_change_pct,
            "sentiment_impact": sentiment_impact,
        }

    def compute_elasticity(self, base_price: float, params: dict) -> float:
        """Compute price elasticity of demand at a given price point."""
        delta = base_price * 0.01
        params_low = {**params, "price": base_price - delta}
        params_high = {**params, "price": base_price + delta}

        X_low = pd.DataFrame([params_low])[self.feature_names]
        X_high = pd.DataFrame([params_high])[self.feature_names]

        demand_low = self.predict(X_low)[0]
        demand_high = self.predict(X_high)[0]

        pct_change_demand = (demand_high - demand_low) / demand_low
        pct_change_price = (2 * delta) / base_price
        return float(pct_change_demand / pct_change_price) if pct_change_price != 0 else 0.0

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=X.index)
        for col in self.feature_names:
            if col in X.columns:
                result[col] = X[col]
            else:
                result[col] = 0.0
        return result
