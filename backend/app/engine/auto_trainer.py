"""Auto-trainer: detects trainable datasets and trains ML models automatically."""

import pandas as pd
from pathlib import Path
from typing import Optional

from app.config import settings
from app.engine.data_analyzer import analyze_dataframe, validate_for_training
from app.engine.feature_builder import build_feature_matrix
from app.engine.simulation import get_simulation_engine
from app.engine.counterfactual import counterfactual_engine


async def auto_train_from_file(file_path: str, user_id: str) -> dict:
    """
    Load a dataset, detect which ML models it can train, and train them.
    Returns a summary dict with trained models and their metrics.
    """
    path = Path(file_path)
    if not path.exists():
        return {"trained": [], "skipped": [], "errors": []}

    try:
        if path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            for enc in ["utf-8", "latin-1", "cp1252"]:
                try:
                    df = pd.read_csv(path, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                return {"trained": [], "skipped": [], "errors": ["Could not decode file"]}
    except Exception as e:
        return {"trained": [], "skipped": [], "errors": [str(e)]}

    if len(df) < 10:
        return {"trained": [], "skipped": [], "errors": ["Dataset too small (< 10 rows)"]}

    analysis = analyze_dataframe(df)
    column_mapping = analysis["column_mapping"]

    engine = get_simulation_engine()
    trained = []
    skipped = []
    errors = []

    for model_name in ["sentiment", "churn", "marketing", "pricing"]:
        try:
            validation = validate_for_training(df, column_mapping, model_name)
            if not validation["valid"]:
                skipped.append({"model": model_name, "reason": validation["issues"][0] if validation["issues"] else "Validation failed"})
                continue

            result = _train_single_model(engine, model_name, df, column_mapping, validation)
            if result:
                trained.append(result)

                # Sync to RAG
                await _sync_to_rag(user_id, model_name, result, file_path, len(df))
        except Exception as e:
            errors.append({"model": model_name, "error": str(e)})

    return {"trained": trained, "skipped": skipped, "errors": errors}


def _train_single_model(engine, model_name: str, df: pd.DataFrame, column_mapping: dict, validation: dict) -> Optional[dict]:
    """Train a single model and return metrics."""
    model = engine.models[model_name]

    if model_name == "sentiment":
        text_col = column_mapping.get("text")
        label_col = column_mapping.get("sentiment")
        if not text_col:
            return None

        # Verify the mapped text column actually has long text content
        # If it looks like IDs (short strings), try to find the real text column
        avg_len = df[text_col].astype(str).str.len().mean()
        if avg_len < 20:
            # Look for a better text column
            candidates = [c for c in df.columns
                          if df[c].dtype == 'object'
                          and df[c].astype(str).str.len().mean() > 20
                          and c != label_col]
            if candidates:
                text_col = candidates[0]

        X = df[[text_col]]
        y = df[label_col] if label_col and label_col in df.columns else None
        metrics = model.train(X, y)

    elif model_name == "churn":
        target_col = column_mapping.get("churn")
        if not target_col or target_col not in df.columns:
            return None

        target_vals = pd.to_numeric(df[target_col], errors="coerce")
        unique_vals = set(target_vals.dropna().unique())
        if not unique_vals <= {0.0, 1.0}:
            return None

        feature_roles = [r for r in model.FEATURE_ROLES if column_mapping.get(r)]
        X = build_feature_matrix(df, column_mapping, feature_roles)
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
        target_col = column_mapping.get("conversion_rate")
        if not target_col or target_col not in df.columns:
            return None

        feature_roles = [r for r in model.FEATURE_ROLES if column_mapping.get(r)]
        X = build_feature_matrix(df, column_mapping, feature_roles)
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
        target_col = column_mapping.get("demand")
        if not target_col or target_col not in df.columns:
            return None

        feature_roles = [r for r in model.FEATURE_ROLES if column_mapping.get(r)]
        X = build_feature_matrix(df, column_mapping, feature_roles)
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
    else:
        return None

    return {"model": model_name, "metrics": metrics}


async def _sync_to_rag(user_id: str, model_name: str, result: dict, dataset_path: str, n_rows: int):
    """Sync training results to the RAG vector database."""
    if not settings.supabase_url:
        return

    try:
        from app.rag.ml_sync import sync_model_to_knowledge_base

        metrics = result.get("metrics", {})
        engine = get_simulation_engine()
        model = engine.models[model_name]

        feature_imps = {}
        if hasattr(model, "model") and hasattr(model.model, "feature_importances_"):
            feature_names = getattr(model, "_feature_names", [])
            for i, imp in enumerate(model.model.feature_importances_):
                if i < len(feature_names):
                    feature_imps[feature_names[i]] = float(imp)

        await sync_model_to_knowledge_base(
            user_id=user_id,
            model_name=model_name,
            metrics=metrics,
            feature_importances=feature_imps,
            dataset_info=f"{Path(dataset_path).name} ({n_rows} rows)",
        )
    except Exception:
        pass
