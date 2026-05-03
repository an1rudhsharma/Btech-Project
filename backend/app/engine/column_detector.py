"""Auto-detect columns from any uploaded business dataset."""

from typing import Optional

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
        "description", "summary", "tweet", "content",
    ],
    "sentiment": ["sentiment", "label", "polarity", "rating", "score"],
    "customer_id": ["customer_id", "user_id", "id", "client_id"],
    "tenure": ["tenure", "months", "duration", "lifetime"],
    "contract": ["contract", "plan", "subscription"],
    "satisfaction": ["satisfaction", "csat", "nps"],
}


def detect_columns(columns: list[str]) -> dict[str, Optional[str]]:
    """Map semantic roles to actual column names in the dataset."""
    detected = {}
    col_lower_map = {c.lower().strip(): c for c in columns}

    for role, patterns in COLUMN_PATTERNS.items():
        found = None
        for pattern in patterns:
            for col_lower, col_original in col_lower_map.items():
                if pattern in col_lower:
                    found = col_original
                    break
            if found:
                break
        detected[role] = found

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
