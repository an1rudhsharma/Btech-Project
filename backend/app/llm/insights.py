"""LLM Insight Generation with Chain-of-Thought reasoning."""

import json
from typing import Optional
from app.llm.prompts import COT_INSIGHT_PROMPT, REPORT_GENERATION_PROMPT


async def generate_insight(
    llm_client,
    predictions: dict,
    shap_values: dict,
    baseline: dict,
    current: dict,
    deltas: dict,
    counterfactuals: dict,
    propagation_trace: dict,
) -> str:
    """Generate a grounded business insight using CoT prompting."""
    prompt = COT_INSIGHT_PROMPT.format(
        predictions=json.dumps(predictions, indent=2, default=str),
        shap_values=json.dumps(shap_values, indent=2, default=str),
        baseline=json.dumps(baseline, indent=2, default=str),
        current=json.dumps(current, indent=2, default=str),
        deltas=json.dumps(deltas, indent=2, default=str),
        counterfactuals=json.dumps(counterfactuals, indent=2, default=str),
        propagation_trace=json.dumps(propagation_trace, indent=2, default=str),
    )

    response = await llm_client.chat(prompt)
    return response


async def generate_report(
    llm_client,
    results: dict,
    scenarios: list[dict],
    metrics: dict,
) -> str:
    """Generate a full business report from simulation results."""
    prompt = REPORT_GENERATION_PROMPT.format(
        results=json.dumps(results, indent=2, default=str),
        scenarios=json.dumps(scenarios, indent=2, default=str),
        metrics=json.dumps(metrics, indent=2, default=str),
    )

    response = await llm_client.chat(prompt)
    return response
