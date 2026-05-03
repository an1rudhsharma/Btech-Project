"""Model training endpoints."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd

from app.config import settings
from app.engine.column_detector import detect_columns
from app.engine.feature_builder import build_feature_matrix
from app.engine.simulation import simulation_engine
from app.engine.counterfactual import counterfactual_engine

router = APIRouter()


class TrainRequest(BaseModel):
    dataset_path: str
    model_name: str  # "churn", "marketing", "pricing", "sentiment"
    target_column: Optional[str] = None
    column_mapping: Optional[dict] = None


@router.post("/train")
async def train_model(request: TrainRequest):
    """Train a specific model on the given dataset."""
    model_name = request.model_name
    if model_name not in simulation_engine.models:
        raise HTTPException(400, f"Unknown model: {model_name}. Valid: churn, marketing, pricing, sentiment")

    try:
        path = request.dataset_path
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
    except Exception as e:
        raise HTTPException(400, f"Could not load dataset: {str(e)}")

    detected = detect_columns(list(df.columns))
    if request.column_mapping:
        detected.update(request.column_mapping)

    model = simulation_engine.models[model_name]

    if model_name == "sentiment":
        text_col = detected.get("text")
        label_col = detected.get("sentiment")
        if not text_col:
            raise HTTPException(400, "No text column found for sentiment model")
        X = df[[text_col]]
        y = df[label_col] if label_col and label_col in df.columns else None
        metrics = model.train(X, y)

    elif model_name == "churn":
        target_col = request.target_column or detected.get("churn")
        if not target_col or target_col not in df.columns:
            raise HTTPException(400, "No churn/target column found")

        feature_roles = [r for r in model.FEATURE_ROLES if detected.get(r)]
        X = build_feature_matrix(df, detected, feature_roles)
        y = df[target_col]
        metrics = model.train(X, y)

        # Setup DiCE for counterfactuals
        try:
            train_df = X.copy()
            train_df["churn"] = y.values
            counterfactual_engine.setup_for_model(
                "churn", train_df, "churn", model.model,
                continuous_features=list(X.columns),
            )
        except Exception:
            pass

    elif model_name == "marketing":
        target_col = request.target_column or detected.get("conversion_rate")
        if not target_col or target_col not in df.columns:
            raise HTTPException(400, "No conversion_rate/target column found")

        feature_roles = [r for r in model.FEATURE_ROLES if detected.get(r)]
        X = build_feature_matrix(df, detected, feature_roles)
        y = df[target_col]
        metrics = model.train(X, y)

        try:
            train_df = X.copy()
            train_df[target_col] = y.values
            counterfactual_engine.setup_for_model(
                "marketing", train_df, target_col, model.model,
                continuous_features=list(X.columns),
            )
        except Exception:
            pass

    elif model_name == "pricing":
        target_col = request.target_column or detected.get("demand")
        if not target_col or target_col not in df.columns:
            raise HTTPException(400, "No demand/target column found")

        feature_roles = [r for r in model.FEATURE_ROLES if detected.get(r)]
        X = build_feature_matrix(df, detected, feature_roles)
        y = df[target_col]
        metrics = model.train(X, y)

        try:
            train_df = X.copy()
            train_df[target_col] = y.values
            counterfactual_engine.setup_for_model(
                "pricing", train_df, target_col, model.model,
                continuous_features=list(X.columns),
            )
        except Exception:
            pass

    return {
        "model": model_name,
        "status": "trained",
        "metrics": metrics,
    }


@router.post("/train/all")
async def train_all_models(dataset_path: str = ""):
    """Train all models on sample data or a specified dataset."""
    if not dataset_path:
        dataset_path = str(settings.data_dir)

    results = {}

    # Train each model on its respective sample data
    sample_map = {
        "churn": "sample_churn.csv",
        "marketing": "sample_marketing.csv",
        "pricing": "sample_pricing.csv",
        "sentiment": "sample_sentiment.csv",
    }

    for model_name, filename in sample_map.items():
        filepath = settings.data_dir / filename
        if filepath.exists():
            try:
                req = TrainRequest(
                    dataset_path=str(filepath),
                    model_name=model_name,
                )
                result = await train_model(req)
                results[model_name] = result
            except Exception as e:
                results[model_name] = {"status": "error", "message": str(e)}
        else:
            results[model_name] = {"status": "skipped", "message": f"{filename} not found"}

    return results
