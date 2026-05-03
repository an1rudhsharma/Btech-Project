"""Safe feature engineering - handles missing columns gracefully."""

import pandas as pd
import numpy as np
from typing import Optional


def safe_get_column(df: pd.DataFrame, col: Optional[str], default: float = 0.0) -> pd.Series:
    """Get a column from dataframe, filling missing values with median or default."""
    if col and col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
        return series.fillna(series.median() if not series.isna().all() else default)
    return pd.Series([default] * len(df), index=df.index)


def build_feature_matrix(
    df: pd.DataFrame,
    detected_columns: dict[str, Optional[str]],
    feature_roles: list[str],
) -> pd.DataFrame:
    """Build a clean feature matrix from detected columns."""
    features = {}
    defaults = {
        "price": 100.0,
        "marketing_spend": 5000.0,
        "num_features": 5.0,
        "usage": 50.0,
        "impressions": 10000.0,
        "clicks": 500.0,
        "tenure": 12.0,
        "satisfaction": 3.0,
        "demand": 100.0,
    }

    for role in feature_roles:
        col = detected_columns.get(role)
        default = defaults.get(role, 0.0)
        features[role] = safe_get_column(df, col, default)

    return pd.DataFrame(features)


def build_single_input(params: dict, feature_roles: list[str]) -> pd.DataFrame:
    """Build a single-row input from simulation parameters."""
    defaults = {
        "price": 100.0,
        "marketing_spend": 5000.0,
        "num_features": 5.0,
        "usage": 50.0,
        "impressions": 10000.0,
        "clicks": 500.0,
        "tenure": 12.0,
        "satisfaction": 3.0,
        "demand": 100.0,
        "sentiment_score": 0.5,
        "predicted_demand": 100.0,
        "predicted_conversion": 0.05,
    }

    row = {}
    for role in feature_roles:
        row[role] = params.get(role, defaults.get(role, 0.0))

    return pd.DataFrame([row])
