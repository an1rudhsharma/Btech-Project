"""Auto-detect columns from any uploaded business dataset."""

import re
from typing import Optional
import pandas as pd

COLUMN_PATTERNS = {
    "churn": ["churn", "left", "attrition", "cancelled", "churned", "exited"],
    "conversion_rate": ["conversion_rate", "conversionrate", "ctr", "click_through"],
    "price": ["price", "unit_price", "selling_price", "list_price", "retail_price",
              "amountmax", "amountmin", "amount_max", "amount_min",
              "cost", "charge", "fee", "mrp", "discount", "cost_per"],
    "marketing_spend": [
        "marketing_spend", "marketing", "ad_spend", "budget", "campaign_cost",
        "advertising", "promotion", "spend",
    ],
    "num_features": ["feature", "num_features", "product_features", "services",
                     "numberofdeviceregistered", "number_of_device"],
    "usage": ["usage", "activity", "sessions", "login", "frequency",
              "hourspend", "hour_spend", "hours_used"],
    "impressions": ["impression", "views", "reach", "exposure"],
    "clicks": ["click", "ctr_raw", "interactions"],
    "conversions": ["converted", "purchase", "signup"],
    "demand": ["demand", "sales", "units_sold", "quantity", "orders", "volume",
               "total_spend", "order_count", "ordercount"],
    "revenue": ["revenue", "income", "earnings", "total_sales", "profit"],
    "text": [
        "text", "review", "comment", "feedback", "message",
        "description", "summary", "tweet", "content", "body",
        "abstract", "title", "note",
    ],
    "sentiment": ["sentiment", "label", "polarity", "rating"],
    "customer_id": ["customer_id", "user_id", "client_id", "row_id", "order_id", "customerid"],
    "tenure": ["tenure", "months", "duration", "lifetime"],
    "contract": ["contract", "plan", "subscription", "license", "contract_length"],
    "satisfaction": ["satisfaction", "csat", "nps", "satisfactionscore", "satisfaction_score"],
    "engagement": ["engagement", "engagement_score"],
}

ID_SUFFIXES = ["id", "_id", "uuid", "code", "key", "hash", "row id", "order id"]


def _normalize_column_name(col: str) -> str:
    """Normalize a column name for matching: handles dot-notation, CamelCase, spaces."""
    # Remove dot-prefix (e.g., "prices.amountMax" -> "amountmax")
    if "." in col:
        col = col.rsplit(".", 1)[-1]
    # CamelCase to snake_case (e.g., "ConversionRate" -> "conversion_rate")
    col = re.sub(r'(?<=[a-z])(?=[A-Z])', '_', col)
    # Replace spaces, dashes, and special chars with underscores
    col = re.sub(r'[\s\-\.]+', '_', col)
    return col.lower().strip().strip('_')


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
    
    Uses a multi-pass approach:
    1. Exact match on lowercased column name
    2. Exact match on normalized name (handles CamelCase, dot-notation)
    3. Substring match on normalized name
    4. (text role) Heuristic: pick longest string column
    
    If a DataFrame is provided, validates text columns have actual long text content.
    """
    detected = {}
    col_lower_map = {c.lower().strip(): c for c in columns}
    col_normalized_map = {_normalize_column_name(c): c for c in columns}
    used_columns = set()

    for role, patterns in COLUMN_PATTERNS.items():
        found = None

        # Pass 1: Exact match on lowercased name
        for pattern in patterns:
            for col_lower, col_original in col_lower_map.items():
                if col_lower == pattern and col_original not in used_columns:
                    found = col_original
                    break
            if found:
                break

        # Pass 2: Exact match on normalized name
        if not found:
            for pattern in patterns:
                for col_norm, col_original in col_normalized_map.items():
                    if col_norm == pattern and col_original not in used_columns:
                        found = col_original
                        break
                if found:
                    break

        # Pass 3: Substring match on normalized name
        if not found:
            for pattern in patterns:
                for col_norm, col_original in col_normalized_map.items():
                    if col_original in used_columns:
                        continue
                    if pattern in col_norm:
                        if role == "text" and _is_id_column(col_norm):
                            continue
                        found = col_original
                        break
                if found:
                    break

        # Pass 4: Substring match on original lowercased name (for dot-notation prefixes)
        if not found:
            for pattern in patterns:
                for col_lower, col_original in col_lower_map.items():
                    if col_original in used_columns:
                        continue
                    if pattern in col_lower:
                        if role == "text" and _is_id_column(col_lower):
                            continue
                        found = col_original
                        break
                if found:
                    break

        # Pass 5 (text role only): Pick the longest string column
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
