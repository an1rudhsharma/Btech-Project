"""Data-first column analysis: infer roles from actual data values, not just names."""

import pandas as pd
import numpy as np
from typing import Optional

from app.engine.column_detector import COLUMN_PATTERNS

ROLE_HEURISTICS = {
    "price": {"dtype": "numeric", "range": (0, 10000), "typical_mean": (10, 1000)},
    "marketing_spend": {"dtype": "numeric", "range": (0, 1000000), "typical_mean": (500, 50000)},
    "usage": {"dtype": "numeric", "range": (0, 100), "typical_mean": (10, 90)},
    "impressions": {"dtype": "numeric", "range": (0, 10000000), "typical_mean": (100, 500000)},
    "clicks": {"dtype": "numeric", "range": (0, 1000000), "typical_mean": (10, 50000)},
    "churn": {"dtype": "binary", "values": {0, 1}},
    "tenure": {"dtype": "numeric", "range": (0, 120), "typical_mean": (1, 60)},
    "satisfaction": {"dtype": "numeric", "range": (0, 10), "typical_mean": (1, 5)},
    "num_features": {"dtype": "numeric", "range": (0, 50), "typical_mean": (1, 20)},
    "text": {"dtype": "string", "avg_length": (10, 5000)},
    "demand": {"dtype": "numeric", "range": (0, 100000), "typical_mean": (10, 10000)},
    "conversion_rate": {"dtype": "numeric", "range": (0, 1), "typical_mean": (0.001, 0.5)},
    "revenue": {"dtype": "numeric", "range": (0, 10000000), "typical_mean": (100, 1000000)},
    "customer_id": {"dtype": "id"},
}


def profile_column(series: pd.Series) -> dict:
    """Compute statistical profile for a single column."""
    profile = {
        "name": series.name,
        "dtype": str(series.dtype),
        "n_rows": len(series),
        "n_null": int(series.isna().sum()),
        "n_unique": int(series.nunique()),
        "unique_ratio": series.nunique() / max(len(series), 1),
    }

    numeric = pd.to_numeric(series, errors="coerce")
    non_null_numeric = numeric.dropna()

    if len(non_null_numeric) > len(series) * 0.5:
        profile["inferred_type"] = "numeric"
        profile["min"] = float(non_null_numeric.min())
        profile["max"] = float(non_null_numeric.max())
        profile["mean"] = float(non_null_numeric.mean())
        profile["std"] = float(non_null_numeric.std()) if len(non_null_numeric) > 1 else 0.0
        profile["numeric_pct"] = len(non_null_numeric) / len(series)
        unique_vals = set(non_null_numeric.unique())
        if unique_vals <= {0.0, 1.0}:
            profile["inferred_type"] = "binary"
    else:
        str_series = series.dropna().astype(str)
        if len(str_series) > 0:
            avg_len = str_series.str.len().mean()
            profile["inferred_type"] = "string"
            profile["avg_str_length"] = float(avg_len)
            profile["samples"] = str_series.head(5).tolist()
        else:
            profile["inferred_type"] = "empty"

    return profile


def score_role_fit(profile: dict, role: str) -> float:
    """Score how well a column's data profile matches a given role (0-1)."""
    heuristic = ROLE_HEURISTICS.get(role)
    if not heuristic:
        return 0.0

    expected_dtype = heuristic.get("dtype")

    if expected_dtype == "string":
        if profile.get("inferred_type") != "string":
            return 0.0
        avg_len = profile.get("avg_str_length", 0)
        lo, hi = heuristic.get("avg_length", (10, 5000))
        if lo <= avg_len <= hi:
            return 0.85
        return 0.2

    if expected_dtype == "binary":
        if profile.get("inferred_type") == "binary":
            return 0.95
        return 0.0

    if expected_dtype == "id":
        if profile.get("unique_ratio", 0) > 0.9:
            return 0.9
        return 0.1

    if expected_dtype == "numeric":
        if profile.get("inferred_type") not in ("numeric", "binary"):
            return 0.0

        score = 0.0
        val_range = heuristic.get("range")
        typical_mean = heuristic.get("typical_mean")

        col_min = profile.get("min", 0)
        col_max = profile.get("max", 0)
        col_mean = profile.get("mean", 0)

        if val_range:
            lo, hi = val_range
            if lo <= col_min and col_max <= hi:
                score += 0.35
            elif col_min >= lo * 0.5 and col_max <= hi * 2:
                score += 0.15

        if typical_mean:
            lo, hi = typical_mean
            if lo <= col_mean <= hi:
                score += 0.35
            elif lo * 0.5 <= col_mean <= hi * 2:
                score += 0.15

        if profile.get("numeric_pct", 0) > 0.9:
            score += 0.1

        # Penalize if the range is extreme relative to what's expected (reduces false positives)
        if val_range and typical_mean:
            expected_range = val_range[1] - val_range[0]
            actual_range = col_max - col_min
            if expected_range > 0 and actual_range > 0:
                range_ratio = actual_range / expected_range
                if range_ratio < 0.01 or range_ratio > 10:
                    score *= 0.5

        return min(score, 1.0)

    return 0.0


def match_by_name(col_name: str) -> Optional[str]:
    """Try to match column name to a role using pattern matching."""
    col_lower = col_name.lower().strip()

    # Skip columns that look like IDs for non-ID roles
    is_id = any(col_lower.endswith(s) for s in ("id", "_id", "uuid", "code", "key", "hash")) or col_lower in ("id", "uid")

    # Pass 1: Exact match
    for role, patterns in COLUMN_PATTERNS.items():
        for pattern in patterns:
            if col_lower == pattern:
                return role

    # Pass 2: Substring match with priority ordering
    # Check more specific roles first (sentiment, churn, etc.) before generic ones (text)
    priority_order = [
        "sentiment", "churn", "conversion_rate", "demand", "revenue",
        "price", "marketing_spend", "impressions", "clicks", "conversions",
        "customer_id", "tenure", "contract", "satisfaction",
        "num_features", "usage", "text",
    ]
    for role in priority_order:
        patterns = COLUMN_PATTERNS.get(role, [])
        if role == "text" and is_id:
            continue
        for pattern in patterns:
            if pattern in col_lower:
                return role

    return None


def analyze_dataframe(df: pd.DataFrame) -> dict:
    """
    Full data-first analysis pipeline.

    Returns:
        {
            "column_mapping": {role: col_name},
            "renamed_columns": {original_name: role_name},
            "rejected_columns": [{"column": str, "reason": str}],
            "accepted_columns": [{"column": str, "role": str, "method": str, "confidence": float}],
            "unmapped_roles": [str],
            "data_issues": [{"column": str, "role": str, "issue": str}],
        }
    """
    profiles = {col: profile_column(df[col]) for col in df.columns}

    column_mapping = {}
    renamed_columns = {}
    accepted_columns = []
    rejected_columns = []
    data_issues = []
    used_columns = set()

    # Pass 1: Match by column name + validate data
    for col in df.columns:
        role = match_by_name(col)
        if role and role not in column_mapping:
            profile = profiles[col]
            heuristic = ROLE_HEURISTICS.get(role)

            if heuristic is None:
                column_mapping[role] = col
                accepted_columns.append({"column": col, "role": role, "method": "name_match", "confidence": 0.8})
                used_columns.add(col)
                continue

            expected_dtype = heuristic.get("dtype")

            if expected_dtype == "string" and profile.get("inferred_type") == "string":
                # Verify text columns actually have long content (not IDs)
                if role == "text":
                    avg_len = df[col].astype(str).str.len().mean()
                    if avg_len < 15:
                        rejected_columns.append({"column": col, "reason": f"Matched 'text' role by name but avg content length is only {avg_len:.0f} chars (looks like IDs)"})
                        continue
                column_mapping[role] = col
                accepted_columns.append({"column": col, "role": role, "method": "name_match", "confidence": 0.95})
                used_columns.add(col)
            elif expected_dtype == "binary" and profile.get("inferred_type") == "binary":
                column_mapping[role] = col
                accepted_columns.append({"column": col, "role": role, "method": "name_match", "confidence": 0.95})
                used_columns.add(col)
            elif expected_dtype == "numeric" and profile.get("inferred_type") in ("numeric", "binary"):
                if profile.get("numeric_pct", 0) >= 0.5:
                    column_mapping[role] = col
                    accepted_columns.append({"column": col, "role": role, "method": "name_match", "confidence": 0.9})
                    used_columns.add(col)
                else:
                    data_issues.append({
                        "column": col,
                        "role": role,
                        "issue": f"Column '{col}' matches role '{role}' by name but contains "
                                 f"{(1 - profile.get('numeric_pct', 0)) * 100:.0f}% non-numeric values",
                    })
                    rejected_columns.append({"column": col, "reason": f"Right name but wrong data: non-numeric values in '{col}'"})
            elif expected_dtype == "id":
                column_mapping[role] = col
                accepted_columns.append({"column": col, "role": role, "method": "name_match", "confidence": 0.8})
                used_columns.add(col)
            else:
                data_issues.append({
                    "column": col,
                    "role": role,
                    "issue": f"Column '{col}' matches role '{role}' by name but data type is "
                             f"'{profile.get('inferred_type')}', expected '{expected_dtype}'",
                })
                rejected_columns.append({"column": col, "reason": f"Right name but wrong data type for role '{role}'"})

    # Pass 2: For unmapped roles, infer from data heuristics (conservative threshold)
    all_roles = set(ROLE_HEURISTICS.keys())
    unmapped_roles = all_roles - set(column_mapping.keys())
    unassigned_cols = [c for c in df.columns if c not in used_columns]

    # Only infer roles when data strongly matches — threshold 0.8 prevents false positives
    INFERENCE_THRESHOLD = 0.8

    for role in list(unmapped_roles):
        best_col = None
        best_score = 0.0

        for col in unassigned_cols:
            score = score_role_fit(profiles[col], role)
            if score > best_score:
                best_score = score
                best_col = col

        if best_col and best_score >= INFERENCE_THRESHOLD:
            column_mapping[role] = best_col
            renamed_columns[best_col] = role
            accepted_columns.append({
                "column": best_col,
                "role": role,
                "method": "data_inference",
                "confidence": round(best_score, 2),
            })
            used_columns.add(best_col)
            unassigned_cols.remove(best_col)
            unmapped_roles.discard(role)

    # Columns that need LLM disambiguation (score between 0.4-0.6)
    ambiguous_columns = []
    for col in unassigned_cols:
        profile = profiles[col]
        if profile.get("inferred_type") == "empty":
            continue
        scores = {role: score_role_fit(profile, role) for role in unmapped_roles}
        top_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2]
        if top_scores and 0.35 <= top_scores[0][1] < 0.6:
            ambiguous_columns.append({
                "column": col,
                "profile": profile,
                "top_candidates": top_scores,
            })

    # Post-processing: validate text column has the longest actual text content
    if column_mapping.get("text"):
        text_col = column_mapping["text"]
        avg_len = df[text_col].astype(str).str.len().mean()
        if avg_len < 30:
            # Find the column with the longest average text
            best_col = None
            best_len = avg_len
            for col in df.columns:
                if col in used_columns and col != text_col:
                    continue
                if df[col].dtype == 'object':
                    col_avg = df[col].astype(str).str.len().mean()
                    if col_avg > best_len:
                        best_len = col_avg
                        best_col = col
            if best_col and best_len > avg_len * 2:
                used_columns.discard(text_col)
                column_mapping["text"] = best_col
                used_columns.add(best_col)

    return {
        "column_mapping": column_mapping,
        "renamed_columns": renamed_columns,
        "rejected_columns": rejected_columns,
        "accepted_columns": accepted_columns,
        "unmapped_roles": list(unmapped_roles),
        "data_issues": data_issues,
        "ambiguous_columns": ambiguous_columns,
        "profiles": {col: {k: v for k, v in p.items() if k != "samples"} for col, p in profiles.items()},
    }


async def resolve_ambiguous_with_llm(ambiguous_columns: list, llm_client) -> dict:
    """Use LLM to classify ambiguous columns."""
    from app.llm.prompts import COLUMN_CLASSIFY_PROMPT

    resolutions = {}
    for item in ambiguous_columns:
        col = item["column"]
        profile = item["profile"]

        prompt = COLUMN_CLASSIFY_PROMPT.format(
            col_name=col,
            dtype=profile.get("inferred_type", "unknown"),
            min_val=profile.get("min", "N/A"),
            max_val=profile.get("max", "N/A"),
            mean_val=profile.get("mean", "N/A"),
            n_unique=profile.get("n_unique", 0),
            n_rows=profile.get("n_rows", 0),
            samples=profile.get("samples", [])[:5],
        )

        result = await llm_client.parse_json(prompt)
        if "error" not in result and result.get("role") != "unknown":
            resolutions[col] = {
                "role": result["role"],
                "confidence": result.get("confidence", 0.5),
                "method": "llm_classification",
            }

    return resolutions


def validate_for_training(df: pd.DataFrame, column_mapping: dict, model_name: str) -> dict:
    """Validate that mapped columns have sufficient valid data for training a specific model."""

    # Only the target column is strictly required; features use what's available
    target_roles = {
        "churn": ["churn"],
        "marketing": ["conversion_rate"],
        "pricing": ["demand"],
        "sentiment": ["text"],
    }
    feature_roles = {
        "churn": ["price", "marketing_spend", "num_features", "usage", "tenure", "satisfaction"],
        "marketing": ["marketing_spend", "impressions", "clicks", "price"],
        "pricing": ["price", "marketing_spend", "usage", "demand"],
        "sentiment": ["text"],
    }

    required = target_roles.get(model_name, [])
    optional = feature_roles.get(model_name, [])
    issues = []
    valid_roles = []

    # Validate required (target) columns
    for role in required:
        col = column_mapping.get(role)
        if col is None:
            issues.append(f"Missing required target column for role '{role}'")
            continue
        if col not in df.columns:
            issues.append(f"Mapped column '{col}' for role '{role}' not found in dataframe")
            continue

        if role == "text":
            non_null = df[col].dropna()
            str_vals = non_null[non_null.apply(lambda x: isinstance(x, str) and len(str(x).strip()) > 0)]
            valid_pct = len(str_vals) / max(len(df), 1)
            if valid_pct < 0.5:
                issues.append(f"Column '{col}' (role: {role}): only {valid_pct*100:.0f}% valid text values")
            else:
                valid_roles.append(role)
        else:
            numeric = pd.to_numeric(df[col], errors="coerce")
            valid_pct = numeric.notna().sum() / max(len(df), 1)
            if valid_pct < 0.8:
                issues.append(f"Column '{col}' (role: {role}): only {valid_pct*100:.0f}% valid numeric values (need >80%)")
            else:
                valid_roles.append(role)

    # Validate optional (feature) columns — just note which are usable
    for role in optional:
        if role in required:
            continue
        col = column_mapping.get(role)
        if col is None or col not in df.columns:
            continue
        if role == "text":
            valid_roles.append(role)
        else:
            numeric = pd.to_numeric(df[col], errors="coerce")
            valid_pct = numeric.notna().sum() / max(len(df), 1)
            if valid_pct >= 0.8:
                valid_roles.append(role)

    return {
        "model": model_name,
        "valid": len(issues) == 0,
        "valid_roles": valid_roles,
        "issues": issues,
        "required_roles": required,
        "available_features": [r for r in optional if r in valid_roles],
    }
