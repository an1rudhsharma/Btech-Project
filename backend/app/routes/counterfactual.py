"""Counterfactual explanation endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd

router = APIRouter()


class CounterfactualRequest(BaseModel):
    model_name: str = "churn"
    scenario: dict
    total_cfs: int = 4
    desired_class: str = "opposite"
    features_to_vary: Optional[list[str]] = None
    permitted_range: Optional[dict] = None


@router.post("/counterfactual")
async def get_counterfactuals(request: CounterfactualRequest):
    """Generate counterfactual explanations for a given scenario.

    Returns actionable recommendations like:
    'Reduce price from $120 to $95 to decrease churn from 78% to 32%'
    """
    from app.engine.counterfactual import counterfactual_engine

    if not counterfactual_engine.is_available(request.model_name):
        raise HTTPException(
            400,
            f"Counterfactuals not available for '{request.model_name}'. "
            "Train the model first to enable counterfactual generation.",
        )

    query_df = pd.DataFrame([request.scenario])

    result = counterfactual_engine.generate_counterfactuals(
        model_name=request.model_name,
        query_instance=query_df,
        total_cfs=request.total_cfs,
        desired_class=request.desired_class,
        features_to_vary=request.features_to_vary,
        permitted_range=request.permitted_range,
    )

    return result
