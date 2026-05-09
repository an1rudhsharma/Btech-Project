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

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
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
    await kb_db.delete_document(user["id"], doc_id)
    return {"status": "deleted"}


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

        return {
            "status": "success",
            "document_id": doc_id,
            "filename": filename,
            "file_type": file_type,
            "chunks": len(chunks),
            "queryable": metadata.get("queryable", False),
        }
    except HTTPException:
        raise
    except Exception as e:
        await kb_db.update_document(doc_id, {"status": "error"})
        raise HTTPException(500, f"Processing failed: {str(e)}")


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
