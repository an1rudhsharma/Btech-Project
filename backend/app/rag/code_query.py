"""Text-to-Pandas pipeline - LLM generates pandas code for structured data queries."""

import pandas as pd
import io
import traceback
from typing import Optional

from app.config import settings

CODE_GEN_SYSTEM = """You are a data analyst assistant. The user has uploaded a CSV/Excel dataset.
Given the schema and a natural language question, generate a Python pandas expression to answer it.

RULES:
- The dataframe variable is named `df`
- Output ONLY the Python code, no explanation
- The code must assign the result to a variable called `result`
- Use only pandas and numpy (already imported as pd and np)
- For display, convert result to string if it's not already
- Handle potential NaN values gracefully
- Never use file I/O, network calls, or subprocess

Example output:
result = df.groupby('month')['revenue'].sum().to_string()
"""


async def query_structured_data(
    query: str,
    dataset_info: dict,
    llm_client,
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

    result = _execute_safely(code, dataset_info)
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


def _execute_safely(code: str, dataset_info: dict) -> Optional[str]:
    """Execute pandas code in a restricted environment."""
    import numpy as np

    forbidden = ["import os", "import sys", "subprocess", "open(", "__import__",
                 "exec(", "eval(", "compile(", "globals(", "locals(",
                 "shutil", "pathlib", "requests", "urllib"]
    for f in forbidden:
        if f in code:
            return f"Code rejected: contains forbidden operation '{f}'"

    try:
        # Reconstruct the dataframe from the stored sample/schema
        # In production, you'd load the actual stored file
        df = pd.DataFrame(dataset_info.get("sample_rows", []))
        if df.empty:
            return "No data available to query"

        local_vars = {"df": df, "pd": pd, "np": np}
        exec(code, {"__builtins__": {}}, local_vars)

        result = local_vars.get("result")
        if result is None:
            return "Code executed but produced no result"

        return str(result)
    except Exception as e:
        return f"Query execution error: {str(e)}"
