"""Auto-detect columns from any uploaded business dataset."""

from typing import Optional
import pandas as pd

COLUMN_PATTERNS = {
    "price": ["price", "cost", "amount", "charge", "fee", "rate", "mrp"],
    "marketing_spend": [
        "marketing", "ad_spend", "budget", "campaign_cost",
        "advertising", "promotion", "spend",
    ],
    "num_features": ["feature", "num_features", "product_features", "services"],
    "usage": ["usage", "activity", "engagement", "sessions", "login", "frequency"],
    "impressions": ["impression", "views", "reach", "exposure"],
    "clicks": ["click", "ctr_raw", "interactions"],
    "conversions": ["conversion", "converted", "purchase", "signup"],
    "churn": ["churn", "left", "attrition", "cancelled", "churned", "exited"],
    "conversion_rate": ["conversion_rate", "ctr", "click_through"],
    "demand": ["demand", "sales", "units_sold", "quantity", "orders"],
    "revenue": ["revenue", "income", "earnings", "total_sales"],
    "text": [
        "text", "review", "comment", "feedback", "message",
        "description", "summary", "tweet", "content", "body",
    ],
    "sentiment": ["sentiment", "label", "polarity", "rating", "score"],
    "customer_id": ["customer_id", "user_id", "id", "client_id"],
    "tenure": ["tenure", "months", "duration", "lifetime"],
    "contract": ["contract", "plan", "subscription"],
    "satisfaction": ["satisfaction", "csat", "nps"],
}

ID_SUFFIXES = ["id", "_id", "uuid", "code", "key", "hash"]


def _is_id_column(col_name: str) -> bool:
    """Check if a column name looks like an identifier."""
    lower = col_name.lower().strip()
    if lower in ("id", "uid", "uuid"):
        return True
    for suffix in ID_SUFFIXES:
        if lower.endswith(suffix):
            return True
    return False


def detect_columns(columns: list[str], df: Optional[pd.DataFrame] = None) -> dict[str, Optional[str]]:
    """Map semantic roles to actual column names in the dataset.
    
    Uses a two-pass approach:
    1. Exact match (column name equals pattern)
    2. Substring match (pattern found in column name, excluding ID columns for text role)
    
    If a DataFrame is provided, validates text columns have actual long text content.
    """
    detected = {}
    col_lower_map = {c.lower().strip(): c for c in columns}
    used_columns = set()

    for role, patterns in COLUMN_PATTERNS.items():
        found = None

        # Pass 1: Exact match (highest priority)
        for pattern in patterns:
            for col_lower, col_original in col_lower_map.items():
                if col_lower == pattern and col_original not in used_columns:
                    found = col_original
                    break
            if found:
                break

        # Pass 2: Substring match (skip ID columns for text role)
        if not found:
            for pattern in patterns:
                for col_lower, col_original in col_lower_map.items():
                    if col_original in used_columns:
                        continue
                    if pattern in col_lower:
                        # For 'text' role, skip columns that look like IDs
                        if role == "text" and _is_id_column(col_lower):
                            continue
                        found = col_original
                        break
                if found:
                    break

        # Pass 3 (text role only): If still no text column found, pick the longest string column
        if not found and role == "text" and df is not None:
            best_col = None
            best_avg_len = 0
            for col in columns:
                if col in used_columns:
                    continue
                if df[col].dtype == 'object':
                    avg_len = df[col].astype(str).str.len().mean()
                    if avg_len > best_avg_len and avg_len > 20:
                        best_avg_len = avg_len
                        best_col = col
            if best_col:
                found = best_col

        if found:
            detected[role] = found
            used_columns.add(found)
        else:
            detected[role] = None

    # Validate text column has actual text content if df is provided
    if df is not None and detected.get("text"):
        text_col = detected["text"]
        avg_len = df[text_col].astype(str).str.len().mean()
        if avg_len < 15:
            # Current text column is too short (probably IDs), find a better one
            for col in columns:
                if col == text_col or col in used_columns:
                    continue
                if df[col].dtype == 'object':
                    col_avg_len = df[col].astype(str).str.len().mean()
                    if col_avg_len > 20:
                        detected["text"] = col
                        break

    return detected


def get_available_features(detected: dict[str, Optional[str]], role: str) -> list[str]:
    """Get list of available columns for a given model role."""
    feature_map = {
        "churn": ["price", "marketing_spend", "num_features", "usage", "tenure", "satisfaction"],
        "marketing": ["marketing_spend", "impressions", "clicks", "price"],
        "pricing": ["price", "marketing_spend", "usage", "demand"],
        "sentiment": ["text"],
    }

    required = feature_map.get(role, [])
    return [detected[f] for f in required if detected.get(f) is not None]
