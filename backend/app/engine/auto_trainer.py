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
            xl = pd.ExcelFile(path)
            best_df = None
            for sheet in xl.sheet_names:
                sheet_df = pd.read_excel(xl, sheet_name=sheet)
                if best_df is None or len(sheet_df) > len(best_df):
                    best_df = sheet_df
            df = best_df if best_df is not None else pd.DataFrame()
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
                # For pricing, allow fallback to price-as-target mode
                if model_name == "pricing" and column_mapping.get("price"):
                    pass  # Let _train_single_model handle the fallback
                else:
                    skipped.append({"model": model_name, "reason": validation["issues"][0] if validation["issues"] else "Validation failed"})
                    continue

            result = _train_single_model(engine, model_name, df, column_mapping, validation)
            if result:
                trained.append(result)
                engine.models[model_name].save()

                # Sync to RAG
                await _sync_to_rag(user_id, model_name, result, file_path, len(df))
            elif model_name == "pricing":
                skipped.append({"model": model_name, "reason": "No suitable target column for pricing"})
        except Exception as e:
            errors.append({"model": model_name, "error": str(e)})

    return {"trained": trained, "skipped": skipped, "errors": errors}


def _get_numeric_features(df: pd.DataFrame, exclude: list = None) -> pd.DataFrame:
    """Get all numeric columns as features, excluding specified columns."""
    exclude = exclude or []
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]
    if not feature_cols:
        return pd.DataFrame()
    result = df[feature_cols].copy()
    result = result.fillna(result.median())
    # Sanitize column names for LightGBM (no special JSON chars)
    import re
    result.columns = [re.sub(r'[.\[\]{}:,"\']', '_', c).strip('_') for c in result.columns]
    return result


def _train_single_model(engine, model_name: str, df: pd.DataFrame, column_mapping: dict, validation: dict) -> Optional[dict]:
    """Train a single model and return metrics."""
    model = engine.models[model_name]

    if model_name == "sentiment":
        text_col = column_mapping.get("text")
        label_col = column_mapping.get("sentiment")
        if not text_col:
            return None

        # Verify the mapped text column actually has natural language text
        avg_len = df[text_col].astype(str).str.len().mean()
        sample_values = df[text_col].dropna().head(20).astype(str)

        def _is_natural_text(series: pd.Series) -> bool:
            """Check if a series contains natural language (not URLs, IDs, etc.)."""
            if series.empty:
                return False
            text_sample = " ".join(series.head(10).tolist())
            url_ratio = sum(1 for v in series if "http" in str(v) or "www." in str(v)) / len(series)
            if url_ratio > 0.3:
                return False
            word_count = text_sample.split()
            avg_word_len = sum(len(w) for w in word_count) / max(len(word_count), 1)
            if avg_word_len > 30:
                return False
            return True

        if avg_len < 30 or not _is_natural_text(sample_values):
            # Look for a better text column with real natural language
            candidates = [c for c in df.columns
                          if df[c].dtype == 'object'
                          and c != label_col
                          and c != text_col]
            found_text = False
            for c in candidates:
                c_avg_len = df[c].astype(str).str.len().mean()
                c_sample = df[c].dropna().head(20).astype(str)
                if c_avg_len > 30 and _is_natural_text(c_sample):
                    text_col = c
                    avg_len = c_avg_len
                    found_text = True
                    break
            if not found_text:
                return None

        # Final guard: skip sentiment if no explicit sentiment/label column exists
        # AND the text column avg length is short (likely product names, not reviews)
        if not label_col or label_col not in df.columns:
            if avg_len < 80:
                return None

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
        if feature_roles:
            X = build_feature_matrix(df, column_mapping, feature_roles)
        else:
            X = _get_numeric_features(df, exclude=[target_col])
        if X.empty or X.shape[1] == 0:
            return None
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
        if feature_roles:
            X = build_feature_matrix(df, column_mapping, feature_roles)
        else:
            X = _get_numeric_features(df, exclude=[target_col])
        if X.empty or X.shape[1] == 0:
            return None
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

        # Fallback: if no demand column, use one price column as target
        # and other numeric columns as features (price prediction mode)
        if not target_col or target_col not in df.columns:
            price_col = column_mapping.get("price")
            if price_col and price_col in df.columns:
                target_col = price_col
            else:
                # Try to find any numeric column that looks like a target
                revenue_col = column_mapping.get("revenue")
                if revenue_col and revenue_col in df.columns:
                    target_col = revenue_col

        if not target_col or target_col not in df.columns:
            return None

        # Ensure target is numeric
        target_vals = pd.to_numeric(df[target_col], errors="coerce")
        if target_vals.notna().sum() / len(df) < 0.5:
            return None

        feature_roles = [r for r in model.FEATURE_ROLES if column_mapping.get(r) and column_mapping.get(r) != target_col]
        if feature_roles:
            X = build_feature_matrix(df, column_mapping, feature_roles)
        else:
            X = _get_numeric_features(df, exclude=[target_col])
        if X.empty or X.shape[1] == 0:
            return None
        y = target_vals.fillna(target_vals.median())
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
