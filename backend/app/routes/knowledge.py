"""Knowledge base endpoints - upload, list, delete documents."""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import Optional
import zipfile
import io

from app.auth.middleware import get_current_user
from app.db import knowledge as kb_db
from app.rag.parser import parse_file, chunk_text
from app.rag.embeddings import embed_batch

router = APIRouter()

MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".csv", ".xlsx", ".xls", ".md", ".log", ".zip"}


@router.get("/knowledge")
async def list_documents(user: dict = Depends(get_current_user)):
    """List all documents in the user's knowledge base."""
    docs = await kb_db.list_documents(user["id"])
    return {"documents": docs}


@router.post("/knowledge/upload")
async def upload_document(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    """Upload and process a document into the knowledge base."""
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    from pathlib import Path
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(400, f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB")

    if ext == ".zip":
        return await _process_zip(file_bytes, file.filename, user["id"])

    return await _process_single_file(file_bytes, file.filename, user["id"])


@router.delete("/knowledge/{doc_id}")
async def delete_document(doc_id: str, user: dict = Depends(get_current_user)):
    """Delete a document and all its chunks from the knowledge base."""
    from app.db.supabase_client import get_admin_client
    client = get_admin_client()

    # Check if this doc trained any models
    doc_result = client.table("documents").select("metadata").eq("id", doc_id).eq("user_id", user["id"]).execute()
    trained_models = []
    if doc_result.data:
        meta = doc_result.data[0].get("metadata", {})
        trained_models = meta.get("trained_models", [])

    await kb_db.delete_document(user["id"], doc_id)

    response = {"status": "deleted"}
    if trained_models:
        response["warning"] = f"Models still active: {', '.join(trained_models)}. Use Reset Model to untrain."
        response["trained_models"] = trained_models
    return response


async def _process_single_file(file_bytes: bytes, filename: str, user_id: str) -> dict:
    """Process a single file: parse, chunk, embed, store."""
    text, file_type, metadata = parse_file(filename, file_bytes)

    if not text.strip():
        raise HTTPException(400, f"Could not extract text from '{filename}'")

    doc = await kb_db.create_document(
        user_id=user_id,
        filename=filename,
        file_type=file_type,
        file_size=len(file_bytes),
        metadata=metadata,
    )
    doc_id = doc["id"]

    # Save raw file for queryable datasets so Text-to-Pandas can load them later
    if metadata.get("queryable"):
        _save_raw_file(user_id, doc_id, filename, file_bytes)

    try:
        chunks = chunk_text(text)
        if not chunks:
            await kb_db.update_document(doc_id, {"status": "error", "chunk_count": 0})
            raise HTTPException(400, f"No text chunks generated from '{filename}'")

        embeddings = embed_batch(chunks)

        chunk_records = [
            {
                "document_id": doc_id,
                "user_id": user_id,
                "content": chunk,
                "embedding": emb,
                "chunk_index": i,
                "metadata": {"filename": filename, "file_type": file_type},
            }
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]

        await kb_db.insert_chunks(chunk_records)
        await kb_db.update_document(doc_id, {"status": "ready", "chunk_count": len(chunks)})

        result = {
            "status": "success",
            "document_id": doc_id,
            "filename": filename,
            "file_type": file_type,
            "chunks": len(chunks),
            "queryable": metadata.get("queryable", False),
            "training": None,
        }

        # Auto-train ML models if this is a queryable dataset (CSV/Excel)
        if metadata.get("queryable"):
            try:
                from app.engine.auto_trainer import auto_train_from_file
                from app.config import settings as app_settings
                file_path = app_settings.data_dir / "knowledge" / user_id / f"{doc_id}_{filename}"
                training_result = await auto_train_from_file(str(file_path), user_id)
                result["training"] = training_result

                # Track which models were trained from this document
                trained_model_names = [t["model"] for t in training_result.get("trained", [])]
                if trained_model_names:
                    updated_meta = {**metadata, "trained_models": trained_model_names}
                    await kb_db.update_document(doc_id, {"metadata": updated_meta})
            except Exception:
                pass

        return result
    except HTTPException:
        raise
    except Exception as e:
        await kb_db.update_document(doc_id, {"status": "error"})
        raise HTTPException(500, f"Processing failed: {str(e)}")


def _save_raw_file(user_id: str, doc_id: str, filename: str, file_bytes: bytes):
    """Save raw file to disk for later Text-to-Pandas queries."""
    from app.config import settings
    user_dir = settings.data_dir / "knowledge" / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    file_path = user_dir / f"{doc_id}_{filename}"
    file_path.write_bytes(file_bytes)


async def _process_zip(file_bytes: bytes, zip_filename: str, user_id: str) -> dict:
    """Extract and process all files from a ZIP archive."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(file_bytes))
    except zipfile.BadZipFile:
        raise HTTPException(400, "Invalid or corrupted ZIP file")

    results = []
    for name in zf.namelist():
        if name.startswith("__MACOSX") or name.startswith("."):
            continue
        from pathlib import Path
        ext = Path(name).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS or ext == ".zip":
            continue
        try:
            inner_bytes = zf.read(name)
            result = await _process_single_file(inner_bytes, name, user_id)
            results.append(result)
        except Exception as e:
            results.append({"filename": name, "status": "error", "error": str(e)})

    return {
        "status": "success",
        "type": "zip",
        "filename": zip_filename,
        "total_files": len(results),
        "successful": sum(1 for r in results if r.get("status") == "success"),
        "results": results,
    }
