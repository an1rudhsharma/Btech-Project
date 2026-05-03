"""DiCE-ML Counterfactual Explanations - actionable what-if recommendations."""

import pandas as pd
import numpy as np
from typing import Optional
import dice_ml

from app.config import settings


class CounterfactualEngine:
    """Generates counterfactual explanations using DiCE-ML.

    Instead of just explaining WHY a prediction happened (SHAP),
    this tells the user WHAT TO CHANGE to get a different outcome.
    """

    def __init__(self):
        self._dice_instances: dict = {}

    def setup_for_model(
        self,
        model_name: str,
        training_data: pd.DataFrame,
        target_col: str,
        model,
        continuous_features: list[str],
        categorical_features: Optional[list[str]] = None,
    ):
        """Initialize DiCE for a specific model."""
        data_df = training_data.copy()

        data_interface = dice_ml.Data(
            dataframe=data_df,
            continuous_features=continuous_features,
            outcome_name=target_col,
        )

        model_interface = dice_ml.Model(model=model, backend="sklearn")

        self._dice_instances[model_name] = {
            "data": data_interface,
            "model": model_interface,
            "dice": dice_ml.Dice(data_interface, model_interface, method="random"),
            "continuous_features": continuous_features,
            "categorical_features": categorical_features or [],
        }

    def generate_counterfactuals(
        self,
        model_name: str,
        query_instance: pd.DataFrame,
        total_cfs: int = 4,
        desired_class: str = "opposite",
        features_to_vary: Optional[list[str]] = None,
        permitted_range: Optional[dict] = None,
    ) -> dict:
        """Generate counterfactual explanations for a given instance.

        Returns actionable recommendations like:
        "Reduce price from $120 to $95 to reduce churn probability from 78% to 32%"
        """
        if model_name not in self._dice_instances:
            return {
                "status": "not_available",
                "message": f"DiCE not configured for {model_name}. Train the model first.",
            }

        dice_info = self._dice_instances[model_name]
        dice_exp = dice_info["dice"]

        try:
            cf_result = dice_exp.generate_counterfactuals(
                query_instances=query_instance,
                total_CFs=total_cfs,
                desired_class=desired_class,
                features_to_vary=features_to_vary or dice_info["continuous_features"],
                permitted_range=permitted_range or {},
            )

            cf_df = cf_result.cf_examples_list[0].final_cfs_df
            if cf_df is None or len(cf_df) == 0:
                return {
                    "status": "no_counterfactuals",
                    "message": "Could not find valid counterfactual scenarios.",
                }

            recommendations = self._format_counterfactuals(
                query_instance, cf_df, model_name
            )

            return {
                "status": "success",
                "original": query_instance.to_dict(orient="records")[0],
                "counterfactuals": recommendations,
                "total_generated": len(recommendations),
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
            }

    def _format_counterfactuals(
        self, original: pd.DataFrame, cfs: pd.DataFrame, model_name: str
    ) -> list[dict]:
        """Format counterfactuals into readable recommendations."""
        recommendations = []
        original_row = original.iloc[0]

        for idx, cf_row in cfs.iterrows():
            changes = []
            for col in original.columns:
                if col in cfs.columns:
                    orig_val = original_row[col]
                    cf_val = cf_row[col]
                    if isinstance(orig_val, (int, float)) and isinstance(cf_val, (int, float)):
                        if abs(orig_val - cf_val) > 0.01:
                            direction = "increase" if cf_val > orig_val else "decrease"
                            changes.append({
                                "feature": col,
                                "from": round(float(orig_val), 2),
                                "to": round(float(cf_val), 2),
                                "direction": direction,
                                "change_pct": round(
                                    (cf_val - orig_val) / orig_val * 100
                                    if orig_val != 0 else 0, 1
                                ),
                            })

            outcome_col = [c for c in cfs.columns if c not in original.columns]
            new_outcome = None
            if outcome_col:
                new_outcome = float(cf_row[outcome_col[0]])

            if changes:
                action_text = "; ".join(
                    f"{c['direction']} {c['feature']} from {c['from']} to {c['to']}"
                    for c in changes
                )
                recommendations.append({
                    "action": action_text,
                    "changes": changes,
                    "new_predicted_outcome": new_outcome,
                    "feasibility": self._score_feasibility(changes),
                })

        recommendations.sort(key=lambda x: x["feasibility"], reverse=True)
        return recommendations

    @staticmethod
    def _score_feasibility(changes: list[dict]) -> float:
        """Score how feasible a counterfactual is (0-1).
        Fewer changes and smaller magnitudes = more feasible.
        """
        if not changes:
            return 0.0
        n_changes = len(changes)
        avg_change = np.mean([abs(c["change_pct"]) for c in changes])
        score = 1.0 / (1.0 + n_changes * 0.3 + avg_change * 0.01)
        return round(score, 3)

    def is_available(self, model_name: str) -> bool:
        return model_name in self._dice_instances


counterfactual_engine = CounterfactualEngine()
