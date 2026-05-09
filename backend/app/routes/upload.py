"""Dataset upload endpoints with data-first analysis, encoding fallback, and ZIP support."""

import os
import io
import zipfile
import tempfile
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from pathlib import Path
import pandas as pd

from app.config import settings
from app.engine.column_detector import detect_columns
from app.engine.data_analyzer import analyze_dataframe, resolve_ambiguous_with_llm

router = APIRouter()

UPLOAD_DIR = settings.data_dir / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".zip"}
DATA_EXTENSIONS = {".csv", ".xlsx", ".xls"}
ENCODINGS_TO_TRY = ["utf-8", "latin-1", "cp1252", "iso-8859-1", "utf-16"]


def read_csv_with_fallback(file_path: Path) -> pd.DataFrame:
    """Try multiple encodings to read a CSV file."""
    errors_collected = []
    for encoding in ENCODINGS_TO_TRY:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError as e:
            errors_collected.append(f"{encoding}: {str(e)[:60]}")
        except Exception as e:
            errors_collected.append(f"{encoding}: {str(e)[:60]}")
    raise ValueError(f"Could not decode CSV with any encoding. Tried: {', '.join(ENCODINGS_TO_TRY)}")


def read_datafile(file_path: Path) -> pd.DataFrame:
    """Read a data file (CSV or Excel) with robust error handling."""
    ext = file_path.suffix.lower()
    if ext == ".csv":
        return read_csv_with_fallback(file_path)
    elif ext in (".xlsx", ".xls"):
        try:
            return pd.read_excel(file_path)
        except Exception as e:
            raise ValueError(f"Could not read Excel file: {str(e)}")
    else:
        raise ValueError(f"Unsupported file type: {ext}")


async def process_single_file(file_path: Path, filename: str) -> dict:
    """Process a single data file through the analysis pipeline."""
    try:
        df = read_datafile(file_path)
    except ValueError as e:
        return {"filename": filename, "status": "error", "error": str(e)}

    if len(df) == 0:
        return {"filename": filename, "status": "error", "error": "File is empty — no rows found"}

    if len(df.columns) == 0:
        return {"filename": filename, "status": "error", "error": "File has no columns"}

    analysis = analyze_dataframe(df)

    # Try LLM disambiguation for ambiguous columns
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

    has_critical_issues = len(analysis["data_issues"]) > 0 and len(analysis["accepted_columns"]) == 0

    if has_critical_issues:
        return {
            "filename": filename,
            "status": "rejected",
            "error": "Dataset rejected — data quality issues found",
            "issues": analysis["data_issues"],
            "rejected_columns": analysis["rejected_columns"],
        }

    # Auto-rename columns if needed and save cleaned version
    ext = file_path.suffix.lower()
    if analysis["renamed_columns"]:
        rename_map = {orig: role for orig, role in analysis["renamed_columns"].items()}
        df_renamed = df.rename(columns=rename_map)
        cleaned_path = UPLOAD_DIR / f"cleaned_{filename}"
        if ext == ".csv":
            df_renamed.to_csv(cleaned_path, index=False)
        else:
            df_renamed.to_excel(cleaned_path, index=False)
        final_path = str(cleaned_path)
    else:
        final_path = str(file_path)

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
        "filename": filename,
        "status": "success",
        "rows": len(df),
        "columns": list(df.columns),
        "detected_mapping": analysis["column_mapping"],
        "renamed_columns": analysis["renamed_columns"],
        "accepted_columns": analysis["accepted_columns"],
        "rejected_columns": analysis["rejected_columns"],
        "data_issues": analysis["data_issues"],
        "trainable_models": trainable,
        "path": final_path,
        "sample_data": df.head(5).fillna("").astype(str).to_dict(orient="records"),
    }


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV, Excel, or ZIP file. ZIP can contain multiple data files."""
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: CSV, XLSX, XLS, ZIP")

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(413, f"File too large ({size_mb:.1f}MB). Max: {settings.max_upload_size_mb}MB")

    # Handle ZIP files — extract and process each data file
    if ext == ".zip":
        try:
            zip_buffer = io.BytesIO(content)
            with zipfile.ZipFile(zip_buffer, 'r') as zf:
                data_files = [
                    name for name in zf.namelist()
                    if Path(name).suffix.lower() in DATA_EXTENSIONS
                    and not name.startswith("__MACOSX")
                    and not Path(name).name.startswith(".")
                ]

                if not data_files:
                    raise HTTPException(400, "ZIP file contains no CSV or Excel files")

                results = []
                for data_filename in data_files:
                    extracted_path = UPLOAD_DIR / Path(data_filename).name
                    with zf.open(data_filename) as zipped_file:
                        with open(extracted_path, "wb") as out_file:
                            out_file.write(zipped_file.read())

                    result = await process_single_file(extracted_path, Path(data_filename).name)
                    results.append(result)

                successful = [r for r in results if r.get("status") == "success"]
                failed = [r for r in results if r.get("status") != "success"]

                return {
                    "filename": file.filename,
                    "type": "zip",
                    "total_files": len(data_files),
                    "successful": len(successful),
                    "failed": len(failed),
                    "results": results,
                }
        except zipfile.BadZipFile:
            raise HTTPException(400, "Invalid ZIP file — could not extract")

    # Single file upload
    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        f.write(content)

    result = await process_single_file(save_path, file.filename)

    if result.get("status") == "error":
        os.remove(save_path)
        raise HTTPException(400, result["error"])

    if result.get("status") == "rejected":
        os.remove(save_path)
        raise HTTPException(400, {
            "error": result["error"],
            "issues": result.get("issues", []),
            "rejected_columns": result.get("rejected_columns", []),
        })

    return result


@router.post("/upload/multiple")
async def upload_multiple_datasets(files: List[UploadFile] = File(...)):
    """Upload multiple data files at once."""
    if not files:
        raise HTTPException(400, "No files provided")

    results = []
    for file in files:
        if not file.filename:
            results.append({"filename": "unknown", "status": "error", "error": "No filename"})
            continue

        ext = Path(file.filename).suffix.lower()
        if ext not in DATA_EXTENSIONS:
            results.append({"filename": file.filename, "status": "error", "error": f"Unsupported type: {ext}"})
            continue

        content = await file.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > settings.max_upload_size_mb:
            results.append({"filename": file.filename, "status": "error", "error": f"Too large ({size_mb:.1f}MB)"})
            continue

        save_path = UPLOAD_DIR / file.filename
        with open(save_path, "wb") as f:
            f.write(content)

        result = await process_single_file(save_path, file.filename)
        if result.get("status") == "error":
            os.remove(save_path)
        results.append(result)

    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") != "success"]

    return {
        "total_files": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "results": results,
    }


@router.get("/upload/datasets")
async def list_datasets():
    """List all available datasets (sample + uploaded)."""
    datasets = []

    for f in settings.data_dir.glob("sample_*.csv"):
        try:
            df = read_csv_with_fallback(f)
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
                df = read_datafile(f)
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
