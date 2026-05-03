"""Simulation API endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


class ScenarioInput(BaseModel):
    price: float = 100.0
    marketing_spend: float = 5000.0
    num_features: float = 5.0
    usage: float = 50.0
    impressions: float = 10000.0
    clicks: float = 500.0
    text: str = "Product is good"
    baseline_price: Optional[float] = None


class ComparisonInput(BaseModel):
    baseline: ScenarioInput = ScenarioInput()
    scenario: ScenarioInput
    text: str = "Product is good"


@router.post("/simulate")
async def run_simulation(scenario: ScenarioInput):
    """Run a full causal simulation with the given scenario parameters."""
    from app.engine.simulation import simulation_engine

    params = scenario.model_dump()
    text = params.pop("text")
    params.pop("baseline_price", None)
    results = simulation_engine.simulate(params, text)
    return results


@router.post("/simulate/compare")
async def run_comparison(input_data: ComparisonInput):
    """Compare baseline vs scenario with causal propagation."""
    from app.engine.simulation import simulation_engine

    baseline_params = input_data.baseline.model_dump()
    baseline_text = baseline_params.pop("text")

    scenario_params = input_data.scenario.model_dump()
    scenario_text = scenario_params.pop("text")

    results = simulation_engine.simulate_comparison(
        baseline_params, scenario_params, input_data.text or scenario_text
    )
    return results


@router.post("/simulate/batch")
async def run_batch_scenarios(scenarios: list[ScenarioInput]):
    """Run multiple scenarios and return results for comparison."""
    from app.engine.simulation import simulation_engine

    results = []
    for s in scenarios:
        params = s.model_dump()
        text = params.pop("text")
        result = simulation_engine.simulate(params, text)
        result["input_scenario"] = params
        results.append(result)

    return {"scenarios": results, "count": len(results)}
