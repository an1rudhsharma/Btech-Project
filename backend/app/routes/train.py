"""Model training endpoints with pre-training data validation."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd

from app.config import settings
from app.engine.column_detector import detect_columns
from app.engine.feature_builder import build_feature_matrix
from app.engine.simulation import simulation_engine
from app.engine.counterfactual import counterfactual_engine
from app.engine.data_analyzer import analyze_dataframe, validate_for_training

router = APIRouter()


class TrainRequest(BaseModel):
    dataset_path: str
    model_name: str
    target_column: Optional[str] = None
    column_mapping: Optional[dict] = None


@router.post("/train")
async def train_model(request: TrainRequest):
    """Train a specific model on the given dataset with pre-training validation."""
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

    if len(df) == 0:
        raise HTTPException(400, "Dataset is empty")

    # Run data-first analysis to get column mapping
    analysis = analyze_dataframe(df)
    detected = analysis["column_mapping"]

    # Allow user-provided mapping to override
    if request.column_mapping:
        detected.update(request.column_mapping)

    # Pre-training validation: ensure required columns have >80% valid data
    validation = validate_for_training(df, detected, model_name)
    if not validation["valid"]:
        raise HTTPException(400, {
            "error": f"Cannot train '{model_name}' model — data validation failed",
            "issues": validation["issues"],
            "required_roles": validation["required_roles"],
            "suggestion": "Upload a dataset with valid numeric data for the required columns, or provide a column_mapping override.",
        })

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
            raise HTTPException(400, "No churn/target column found. Need a binary (0/1) column indicating customer churn.")

        # Validate target column is binary
        target_vals = pd.to_numeric(df[target_col], errors="coerce")
        valid_pct = target_vals.notna().sum() / len(df)
        if valid_pct < 0.8:
            raise HTTPException(400, f"Target column '{target_col}' has only {valid_pct*100:.0f}% valid numeric values (need >80%)")

        unique_vals = set(target_vals.dropna().unique())
        if not unique_vals <= {0.0, 1.0}:
            raise HTTPException(400, f"Churn target column must be binary (0/1). Found values: {sorted(list(unique_vals))[:10]}")

        feature_roles = [r for r in model.FEATURE_ROLES if detected.get(r)]
        X = build_feature_matrix(df, detected, feature_roles)
        y = df[target_col]
        metrics = model.train(X, y)

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
            raise HTTPException(400, "No conversion_rate/target column found. Need a numeric column for conversion metric.")

        target_vals = pd.to_numeric(df[target_col], errors="coerce")
        valid_pct = target_vals.notna().sum() / len(df)
        if valid_pct < 0.8:
            raise HTTPException(400, f"Target column '{target_col}' has only {valid_pct*100:.0f}% valid numeric values (need >80%)")

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
            raise HTTPException(400, "No demand/target column found. Need a numeric column for demand/sales.")

        target_vals = pd.to_numeric(df[target_col], errors="coerce")
        valid_pct = target_vals.notna().sum() / len(df)
        if valid_pct < 0.8:
            raise HTTPException(400, f"Target column '{target_col}' has only {valid_pct*100:.0f}% valid numeric values (need >80%)")

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

    # Sync model summary to RAG knowledge base (if Supabase is configured)
    if settings.supabase_url:
        try:
            from app.rag.ml_sync import sync_model_to_knowledge_base
            feature_imps = {}
            if hasattr(model, "model") and hasattr(model.model, "feature_importances_"):
                cols = list(X.columns) if model_name != "sentiment" else []
                for i, imp in enumerate(model.model.feature_importances_):
                    if i < len(cols):
                        feature_imps[cols[i]] = float(imp)
            await sync_model_to_knowledge_base(
                user_id="system",
                model_name=model_name,
                metrics=metrics,
                feature_importances=feature_imps,
                dataset_info=f"{request.dataset_path} ({len(df)} rows)",
            )
        except Exception:
            pass

    return {
        "model": model_name,
        "status": "trained",
        "metrics": metrics,
        "columns_used": {r: detected.get(r) for r in validation["valid_roles"]},
    }


@router.post("/train/all")
async def train_all_models(dataset_path: str = ""):
    """Train all models on sample data or a specified dataset."""
    if not dataset_path:
        dataset_path = str(settings.data_dir)

    results = {}

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
