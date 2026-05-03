"""Dataset upload endpoints."""

import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import pandas as pd

from app.config import settings
from app.engine.column_detector import detect_columns

router = APIRouter()

UPLOAD_DIR = settings.data_dir / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV or Excel dataset for model training."""
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

    # Parse and detect columns
    try:
        if ext == ".csv":
            df = pd.read_csv(save_path)
        else:
            df = pd.read_excel(save_path)
    except Exception as e:
        os.remove(save_path)
        raise HTTPException(400, f"Could not parse file: {str(e)}")

    detected = detect_columns(list(df.columns))

    return {
        "filename": file.filename,
        "rows": len(df),
        "columns": list(df.columns),
        "detected_mapping": detected,
        "sample_data": df.head(5).to_dict(orient="records"),
        "path": str(save_path),
    }


@router.get("/upload/datasets")
async def list_datasets():
    """List all available datasets (sample + uploaded)."""
    datasets = []

    # Sample datasets
    for f in settings.data_dir.glob("sample_*.csv"):
        df = pd.read_csv(f)
        datasets.append({
            "name": f.stem,
            "path": str(f),
            "rows": len(df),
            "columns": list(df.columns),
            "type": "sample",
        })

    # Uploaded datasets
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
