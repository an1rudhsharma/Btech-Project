"""Causal Simulation Engine - runs models in dependency order with signal propagation."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Optional

from app.config import settings
from app.models.churn import ChurnModel
from app.models.sentiment import SentimentModel
from app.models.marketing import MarketingModel
from app.models.pricing import PricingModel
from app.engine.causal_graph import get_execution_order, CAUSAL_GRAPH


class SimulationEngine:
    """Orchestrates all models with causal signal propagation.

    Execution order: Pricing -> Marketing -> Sentiment -> Churn
    Each model's output feeds as input to downstream models.
    """

    def __init__(self):
        model_dir = settings.model_dir
        self.models = {
            "pricing": PricingModel(model_dir),
            "marketing": MarketingModel(model_dir),
            "sentiment": SentimentModel(model_dir),
            "churn": ChurnModel(model_dir),
        }
        self._load_models()

    def _load_models(self):
        """Try to load pre-trained models from disk."""
        for model in self.models.values():
            model.load()

    def get_model_status(self) -> dict[str, dict]:
        """Return training status of all models."""
        return {
            name: {
                "trained": model.is_trained,
                "metrics": model.training_metrics,
            }
            for name, model in self.models.items()
        }

    def simulate(self, scenario: dict, text: str = "Product is good") -> dict:
        """Run a full causal simulation.

        Args:
            scenario: Dict with keys like price, marketing_spend, num_features, etc.
            text: Customer review text for sentiment analysis.

        Returns:
            Dict with predictions from all models + propagation trace.
        """
        context = {}
        results = {}
        execution_order = get_execution_order()

        for model_name in execution_order:
            if not self.models[model_name].is_trained and model_name != "sentiment":
                results[model_name] = {"status": "not_trained"}
                continue

            if model_name == "pricing":
                results["pricing"] = self._run_pricing(scenario, context)
            elif model_name == "marketing":
                results["marketing"] = self._run_marketing(scenario, context)
            elif model_name == "sentiment":
                results["sentiment"] = self._run_sentiment(scenario, context, text)
            elif model_name == "churn":
                results["churn"] = self._run_churn(scenario, context)

        results["propagation_trace"] = context
        results["scenario_input"] = scenario
        return results

    def simulate_comparison(
        self, baseline: dict, scenario: dict, text: str = "Product is good"
    ) -> dict:
        """Run baseline vs scenario comparison."""
        baseline_results = self.simulate(baseline, text)
        scenario_results = self.simulate(scenario, text)

        deltas = {}
        for model_name in ["pricing", "marketing", "sentiment", "churn"]:
            if model_name in baseline_results and model_name in scenario_results:
                base_vals = baseline_results[model_name]
                scen_vals = scenario_results[model_name]
                if isinstance(base_vals, dict) and isinstance(scen_vals, dict):
                    deltas[model_name] = {
                        k: scen_vals.get(k, 0) - base_vals.get(k, 0)
                        for k in base_vals
                        if isinstance(base_vals.get(k), (int, float))
                    }

        return {
            "baseline": baseline_results,
            "scenario": scenario_results,
            "deltas": deltas,
        }

    def _run_pricing(self, scenario: dict, context: dict) -> dict:
        """Execute pricing model and propagate outputs."""
        pricing_model = self.models["pricing"]
        features = {role: scenario.get(role, 0) for role in pricing_model.FEATURE_ROLES}
        X = pd.DataFrame([features])

        baseline_price = scenario.get("baseline_price") or scenario.get("price", 100)
        result = pricing_model.predict_with_impact(X, baseline_price)

        context["predicted_demand"] = float(result["demand"][0])
        context["price_change_pct"] = float(result["price_change_pct"][0])
        context["sentiment_impact"] = float(result["sentiment_impact"][0])
        context["revenue"] = float(result["revenue"][0])

        X_aligned = pricing_model._align_features(X.copy())
        shap_summary = pricing_model.get_shap_summary(X_aligned)

        return {
            "demand": context["predicted_demand"],
            "revenue": context["revenue"],
            "price_change_pct": context["price_change_pct"],
            "elasticity": pricing_model.compute_elasticity(
                scenario.get("price", 100), features
            ),
            "shap_drivers": shap_summary,
        }

    def _run_marketing(self, scenario: dict, context: dict) -> dict:
        """Execute marketing model and propagate outputs."""
        marketing_model = self.models["marketing"]
        features = {role: scenario.get(role, 0) for role in marketing_model.FEATURE_ROLES}
        X = pd.DataFrame([features])

        result = marketing_model.predict_with_engagement(X)

        context["predicted_conversion"] = float(result["conversion"][0])
        context["marketing_effect"] = float(result["engagement"][0])

        X_aligned = marketing_model._align_features(X.copy())
        shap_summary = marketing_model.get_shap_summary(X_aligned)

        return {
            "conversion_rate": context["predicted_conversion"],
            "engagement": context["marketing_effect"],
            "shap_drivers": shap_summary,
        }

    def _run_sentiment(self, scenario: dict, context: dict, text: str) -> dict:
        """Execute sentiment model using propagated context."""
        sentiment_model = self.models["sentiment"]

        sentiment_context = {
            "price_change_pct": context.get("price_change_pct", 0.0),
            "marketing_intensity": context.get("marketing_effect", 1.0),
        }

        score = sentiment_model.predict_single(text, context=sentiment_context)
        context["sentiment_score"] = score

        return {
            "sentiment_score": score,
            "label": "positive" if score > 0.5 else "negative",
            "context_adjustments": sentiment_context,
        }

    def _run_churn(self, scenario: dict, context: dict) -> dict:
        """Execute churn model using ALL propagated signals."""
        churn_model = self.models["churn"]

        features = {role: scenario.get(role, 0) for role in churn_model.FEATURE_ROLES}
        features["sentiment_score"] = context.get("sentiment_score", 0.5)
        features["predicted_demand"] = context.get("predicted_demand", 100)
        features["predicted_conversion"] = context.get("predicted_conversion", 0.05)

        X = pd.DataFrame([features])
        X_aligned = churn_model._align_features(X)
        proba = churn_model.predict_proba(X)
        churn_prob = float(proba[0][1]) if proba is not None else 0.0

        shap_summary = churn_model.get_shap_summary(X_aligned)

        return {
            "churn_probability": churn_prob,
            "risk_level": "high" if churn_prob > 0.7 else "medium" if churn_prob > 0.4 else "low",
            "shap_drivers": shap_summary,
        }

    def get_shap_for_scenario(self, scenario: dict) -> dict:
        """Get SHAP explanations for all models for a given scenario."""
        explanations = {}
        for name, model in self.models.items():
            if model.is_trained and name != "sentiment":
                features = {role: scenario.get(role, 0) for role in model.feature_names}
                X = pd.DataFrame([features])
                explanations[name] = model.get_shap_summary(X)
        return explanations


_engine_instance: Optional[SimulationEngine] = None


def get_simulation_engine() -> SimulationEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = SimulationEngine()
    return _engine_instance


# Backward compat property-like access
class _LazyEngine:
    def __getattr__(self, name):
        return getattr(get_simulation_engine(), name)


simulation_engine = _LazyEngine()
