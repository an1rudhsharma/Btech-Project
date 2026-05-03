"""Dataset upload endpoints with data-first analysis."""

import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import pandas as pd

from app.config import settings
from app.engine.column_detector import detect_columns
from app.engine.data_analyzer import analyze_dataframe, resolve_ambiguous_with_llm

router = APIRouter()

UPLOAD_DIR = settings.data_dir / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV or Excel dataset — analyze data first, auto-rename columns, reject bad data."""
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in (".csv", ".xlsx", ".xls"):
        raise HTTPException(400, "Only CSV and Excel files are supported")

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(413, f"File too large. Max: {settings.max_upload_size_mb}MB")

    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        f.write(content)

    try:
        if ext == ".csv":
            df = pd.read_csv(save_path)
        else:
            df = pd.read_excel(save_path)
    except Exception as e:
        os.remove(save_path)
        raise HTTPException(400, f"Could not parse file: {str(e)}")

    if len(df) == 0:
        os.remove(save_path)
        raise HTTPException(400, "File is empty — no rows found")

    # Run data-first analysis
    analysis = analyze_dataframe(df)

    # Try LLM disambiguation for ambiguous columns (if Groq key is available)
    llm_resolved = {}
    if analysis["ambiguous_columns"] and settings.groq_api_key:
        try:
            from app.llm.orchestrator import orchestrator
            llm_resolved = await resolve_ambiguous_with_llm(
                analysis["ambiguous_columns"], orchestrator.llm
            )
            for col, info in llm_resolved.items():
                role = info["role"]
                if role not in analysis["column_mapping"]:
                    analysis["column_mapping"][role] = col
                    analysis["renamed_columns"][col] = role
                    analysis["accepted_columns"].append({
                        "column": col,
                        "role": role,
                        "method": "llm_classification",
                        "confidence": info["confidence"],
                    })
        except Exception:
            pass

    # If there are data issues (right name, wrong data), flag them clearly
    has_critical_issues = len(analysis["data_issues"]) > 0 and len(analysis["accepted_columns"]) == 0

    if has_critical_issues:
        os.remove(save_path)
        raise HTTPException(400, {
            "error": "Dataset rejected — data quality issues found",
            "issues": analysis["data_issues"],
            "rejected_columns": analysis["rejected_columns"],
        })

    # Auto-rename columns in the saved file if needed
    if analysis["renamed_columns"]:
        rename_map = {orig: role for orig, role in analysis["renamed_columns"].items()}
        df_renamed = df.rename(columns=rename_map)
        cleaned_path = UPLOAD_DIR / f"cleaned_{file.filename}"
        if ext == ".csv":
            df_renamed.to_csv(cleaned_path, index=False)
        else:
            df_renamed.to_excel(cleaned_path, index=False)
        final_path = str(cleaned_path)
    else:
        final_path = str(save_path)

    # Determine which models can be trained
    trainable = []
    for model in ["churn", "marketing", "pricing", "sentiment"]:
        model_roles = {
            "churn": ["churn"],
            "marketing": ["conversion_rate", "marketing_spend"],
            "pricing": ["demand", "price"],
            "sentiment": ["text"],
        }
        required = model_roles.get(model, [])
        if any(r in analysis["column_mapping"] for r in required):
            trainable.append(model)

    return {
        "filename": file.filename,
        "rows": len(df),
        "columns": list(df.columns),
        "detected_mapping": analysis["column_mapping"],
        "renamed_columns": analysis["renamed_columns"],
        "accepted_columns": analysis["accepted_columns"],
        "rejected_columns": analysis["rejected_columns"],
        "data_issues": analysis["data_issues"],
        "trainable_models": trainable,
        "path": final_path,
        "sample_data": df.head(5).to_dict(orient="records"),
    }


@router.get("/upload/datasets")
async def list_datasets():
    """List all available datasets (sample + uploaded)."""
    datasets = []

    for f in settings.data_dir.glob("sample_*.csv"):
        try:
            df = pd.read_csv(f)
            datasets.append({
                "name": f.stem,
                "path": str(f),
                "rows": len(df),
                "columns": list(df.columns),
                "type": "sample",
            })
        except Exception:
            pass

    for f in UPLOAD_DIR.glob("*"):
        if f.suffix in (".csv", ".xlsx", ".xls"):
            try:
                df = pd.read_csv(f) if f.suffix == ".csv" else pd.read_excel(f)
                datasets.append({
                    "name": f.stem,
                    "path": str(f),
                    "rows": len(df),
                    "columns": list(df.columns),
                    "type": "uploaded",
                })
            except Exception:
                pass

    return {"datasets": datasets}
