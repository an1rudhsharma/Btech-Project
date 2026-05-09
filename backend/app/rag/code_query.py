"""Text-to-Pandas pipeline - LLM generates pandas code for structured data queries."""

import pandas as pd
import io
import traceback
from pathlib import Path
from typing import Optional

from app.config import settings

CODE_GEN_SYSTEM = """You are a data analyst assistant. The user has uploaded a CSV/Excel dataset.
Given the schema and a natural language question, generate a Python pandas expression to answer it.

RULES:
- The dataframe variable is named `df`
- Output ONLY the Python code, no explanation
- The code must assign the result to a variable called `result`
- Use only pandas, numpy, and collections.Counter (already imported as pd, np, Counter)
- For display, convert result to string if it's not already
- Handle potential NaN values gracefully
- Never use file I/O, network calls, or subprocess
- For value_counts or groupby, show top 10 at most unless asked otherwise
- Always include actual numbers/counts, not just column names
- For text analysis questions (common words, themes, drivers), split text into words and find frequencies
- When analyzing sentiment drivers, compare word frequencies between positive and negative groups

Example output:
result = df.groupby('month')['revenue'].sum().to_string()

Example for text analysis:
from collections import Counter
negative = df[df['sentiment']=='negative']['text'].dropna().str.lower().str.split().explode()
neg_words = Counter(negative).most_common(20)
result = 'Top negative words: ' + str(neg_words)
"""


def _load_dataset(dataset_info: dict, user_id: str) -> Optional[pd.DataFrame]:
    """Load the actual dataset file from disk."""
    doc_id = dataset_info.get("doc_id", "")
    filename = dataset_info.get("filename", "")

    user_dir = settings.data_dir / "knowledge" / user_id
    file_path = user_dir / f"{doc_id}_{filename}"

    if not file_path.exists():
        # Fallback: try to find any file matching the doc_id
        for f in user_dir.glob(f"{doc_id}_*"):
            file_path = f
            break

    if not file_path.exists():
        return None

    ext = Path(filename).suffix.lower()
    for encoding in ["utf-8", "latin-1", "cp1252"]:
        try:
            if ext in (".xlsx", ".xls"):
                return pd.read_excel(file_path)
            else:
                return pd.read_csv(file_path, encoding=encoding)
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue

    return None


async def query_structured_data(
    query: str,
    dataset_info: dict,
    llm_client,
    user_id: str = "",
) -> Optional[str]:
    """
    Use LLM to generate pandas code, execute it against the dataset.
    Returns the result as a string, or None if it can't be answered.
    """
    schema_desc = f"""Dataset: {dataset_info['filename']}
Shape: {dataset_info['shape'][0]} rows x {dataset_info['shape'][1]} columns
Columns: {', '.join(dataset_info['columns'])}
Data types: {dataset_info['dtypes']}
Sample rows (first 3):
{_format_sample_rows(dataset_info.get('sample_rows', []))}"""

    prompt = f"""{schema_desc}

User question: {query}

Generate pandas code to answer this. Assign the answer to `result`."""

    code_response = await llm_client.chat(prompt, system=CODE_GEN_SYSTEM)

    code = _clean_code(code_response)
    if not code:
        return None

    result = _execute_safely(code, dataset_info, user_id)
    return result


def _format_sample_rows(rows: list[dict]) -> str:
    if not rows:
        return "No sample data available"
    df = pd.DataFrame(rows)
    return df.to_string(index=False)


def _clean_code(response: str) -> str:
    """Extract clean Python code from LLM response."""
    code = response.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        lines = lines[1:]  # Remove opening ```python
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)
    return code.strip()


def _execute_safely(code: str, dataset_info: dict, user_id: str = "") -> Optional[str]:
    """Execute pandas code in a restricted environment."""
    import numpy as np
    from collections import Counter

    forbidden = ["import os", "import sys", "subprocess", "open(", "__import__",
                 "exec(", "eval(", "compile(", "globals(", "locals(",
                 "shutil", "pathlib", "requests", "urllib"]
    for f in forbidden:
        if f in code:
            return f"Code rejected: contains forbidden operation '{f}'"

    try:
        # Try to load the actual full dataset from disk
        df = _load_dataset(dataset_info, user_id) if user_id else None

        # Fallback to sample rows if file not found
        if df is None or df.empty:
            df = pd.DataFrame(dataset_info.get("sample_rows", []))
            if df.empty:
                return "No data available to query"

        local_vars = {"df": df, "pd": pd, "np": np, "Counter": Counter}
        exec(code, {"__builtins__": {"len": len, "str": str, "int": int, "float": float, "list": list, "dict": dict, "tuple": tuple, "set": set, "range": range, "enumerate": enumerate, "zip": zip, "sorted": sorted, "min": min, "max": max, "sum": sum, "abs": abs, "round": round, "print": print, "isinstance": isinstance, "type": type, "True": True, "False": False, "None": None}}, local_vars)

        result = local_vars.get("result")
        if result is None:
            return "Code executed but produced no result"

        return str(result)[:3000]
    except Exception as e:
        return f"Query execution error: {str(e)}"
