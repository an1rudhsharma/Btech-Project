"""LLM Orchestration Layer - routes queries through parsing, simulation, and insight generation."""

import json
from typing import Optional

from app.config import settings
from app.llm.prompts import INTENT_PARSE_PROMPT, INTENT_CLASSIFY_PROMPT
from app.llm.insights import generate_insight, generate_report
from app.engine.simulation import simulation_engine
from app.engine.counterfactual import counterfactual_engine


class LLMClient:
    """Wrapper around Groq API for LLM calls."""

    def __init__(self):
        self._client = None

    def _get_client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=settings.groq_api_key)
        return self._client

    async def chat(self, prompt: str, system: str = "You are a business analytics AI.") -> str:
        """Send a message to the LLM and get a response."""
        try:
            client = self._get_client()
            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"LLM Error: {str(e)}. Please check your GROQ_API_KEY in .env file."

    async def parse_json(self, prompt: str) -> dict:
        """Send a message expecting JSON response."""
        response = await self.chat(prompt, system="You are a JSON parser. Respond ONLY with valid JSON.")
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"error": "Failed to parse LLM response as JSON", "raw": response}


class Orchestrator:
    """Main orchestration layer connecting LLM, simulation engine, and counterfactuals."""

    def __init__(self):
        self.llm = LLMClient()

    async def process_natural_language_query(self, query: str) -> dict:
        """Full pipeline: classify -> parse query -> simulate -> explain."""

        # Step 0: Classify intent — is this a business query or unrelated?
        classify_prompt = INTENT_CLASSIFY_PROMPT.format(query=query)
        classification = await self.llm.parse_json(classify_prompt)

        if classification.get("category") == "unrelated":
            answer = await self.llm.chat(
                query,
                system=(
                    "You are a helpful AI assistant that specializes in business simulation and analytics. "
                    "The user asked something unrelated to business data. Answer their question briefly and helpfully, "
                    "then remind them that you specialize in business decision simulation — pricing, churn prediction, "
                    "marketing analysis, and sentiment analysis. Suggest a relevant business question they could ask."
                ),
            )
            return {"query": query, "insight": answer, "predictions": None, "category": "unrelated"}

        # Step 1: Parse user intent
        parse_prompt = INTENT_PARSE_PROMPT.format(query=query)
        intent = await self.llm.parse_json(parse_prompt)

        if "error" in intent:
            return {"error": intent["error"], "raw_response": intent.get("raw")}

        params = intent.get("parameters", {})
        text = intent.get("text", "Product is good")
        goal = intent.get("goal", "general_analysis")

        scenario = {k: v for k, v in params.items() if v is not None}
        # Fill defaults for missing params
        defaults = {
            "price": 100, "marketing_spend": 5000, "num_features": 5,
            "usage": 50, "impressions": 10000, "clicks": 500,
        }
        for k, v in defaults.items():
            scenario.setdefault(k, v)

        # Step 2: Run causal simulation
        if intent.get("comparison_mode"):
            baseline = {**defaults, **intent.get("baseline_overrides", {})}
            comparison = simulation_engine.simulate_comparison(baseline, scenario, text)
            results = comparison["scenario"]
            baseline_results = comparison["baseline"]
            deltas = comparison["deltas"]
        else:
            results = simulation_engine.simulate(scenario, text)
            baseline_results = simulation_engine.simulate(defaults, "Product is good")
            deltas = self._compute_deltas(baseline_results, results)

        # Step 3: Get SHAP explanations
        shap_values = simulation_engine.get_shap_for_scenario(scenario)

        # Step 4: Generate counterfactuals if relevant
        counterfactuals = {}
        if goal in ("reduce_churn", "general_analysis") and counterfactual_engine.is_available("churn"):
            import pandas as pd
            query_df = pd.DataFrame([scenario])
            counterfactuals = counterfactual_engine.generate_counterfactuals(
                "churn", query_df, total_cfs=3, desired_class="opposite"
            )

        # Step 5: Generate CoT insight
        insight = await generate_insight(
            self.llm,
            predictions=results,
            shap_values=shap_values,
            baseline=baseline_results,
            current=results,
            deltas=deltas,
            counterfactuals=counterfactuals,
            propagation_trace=results.get("propagation_trace", {}),
        )

        return {
            "query": query,
            "parsed_intent": intent,
            "predictions": results,
            "insight": insight,
            "counterfactuals": counterfactuals,
            "shap_explanations": shap_values,
        }

    async def generate_business_report(self, scenarios: list[dict]) -> str:
        """Generate a full report comparing multiple scenarios."""
        all_results = []
        for s in scenarios:
            result = simulation_engine.simulate(s.get("params", {}), s.get("text", "Product is good"))
            all_results.append({"scenario": s.get("name", "unnamed"), "results": result})

        metrics = simulation_engine.get_model_status()
        return await generate_report(self.llm, all_results, scenarios, metrics)

    @staticmethod
    def _compute_deltas(baseline: dict, current: dict) -> dict:
        """Compute differences between baseline and current results."""
        deltas = {}
        for key in ["pricing", "marketing", "sentiment", "churn"]:
            if key in baseline and key in current:
                b = baseline[key]
                c = current[key]
                if isinstance(b, dict) and isinstance(c, dict):
                    deltas[key] = {
                        k: c.get(k, 0) - b.get(k, 0)
                        for k in b
                        if isinstance(b.get(k), (int, float))
                    }
        return deltas


orchestrator = Orchestrator()
