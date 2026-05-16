"""Knowledge base persistence - documents and chunks."""

from typing import Optional
from app.db.supabase_client import get_admin_client


async def create_document(user_id: str, filename: str, file_type: str, file_size: int, metadata: dict = None) -> dict:
    client = get_admin_client()
    result = client.table("documents").insert({
        "user_id": user_id,
        "filename": filename,
        "file_type": file_type,
        "file_size_bytes": file_size,
        "metadata": metadata or {},
        "status": "processing",
    }).execute()
    return result.data[0] if result.data else {}


async def update_document(doc_id: str, updates: dict) -> dict:
    client = get_admin_client()
    result = client.table("documents").update(updates).eq("id", doc_id).execute()
    return result.data[0] if result.data else {}


async def list_documents(user_id: str) -> list:
    client = get_admin_client()
    result = (
        client.table("documents")
        .select("id, filename, file_type, file_size_bytes, chunk_count, status, metadata, uploaded_at")
        .eq("user_id", user_id)
        .neq("file_type", "ml_summary")
        .order("uploaded_at", desc=True)
        .execute()
    )
    return result.data or []


async def delete_document(user_id: str, doc_id: str) -> bool:
    """Delete a document, its chunks, and raw file from disk."""
    client = get_admin_client()
    # Delete all chunks for this document
    client.table("document_chunks").delete().eq("document_id", doc_id).eq("user_id", user_id).execute()
    # Delete the document record
    client.table("documents").delete().eq("id", doc_id).eq("user_id", user_id).execute()
    # Delete raw file from disk
    _delete_raw_file(user_id, doc_id)
    return True


def _delete_raw_file(user_id: str, doc_id: str):
    """Remove the saved raw file from disk."""
    from app.config import settings
    user_dir = settings.data_dir / "knowledge" / user_id
    if not user_dir.exists():
        return
    for f in user_dir.glob(f"{doc_id}_*"):
        try:
            f.unlink()
        except OSError:
            pass


async def insert_chunks(chunks: list[dict]) -> int:
    """Batch insert document chunks with embeddings."""
    if not chunks:
        return 0
    client = get_admin_client()
    client.table("document_chunks").insert(chunks).execute()
    return len(chunks)


async def search_chunks(user_id: str, query_embedding: list[float], top_k: int = 8,
                        threshold: float = 0.3, document_ids: list[str] = None) -> list:
    """Vector similarity search using the match_documents RPC function.
    
    Args:
        user_id: Scope search to this user's documents
        query_embedding: The query vector
        top_k: Maximum number of results
        threshold: Minimum similarity score
        document_ids: Optional list of document IDs to restrict search to
    """
    client = get_admin_client()
    params = {
        "query_embedding": query_embedding,
        "match_user_id": user_id,
        "match_count": top_k,
        "match_threshold": threshold,
    }
    if document_ids:
        params["filter_document_ids"] = document_ids

    try:
        result = client.rpc("match_documents", params).execute()
        return result.data or []
    except Exception:
        # Fallback without document_ids filter if RPC doesn't support it
        fallback_params = {
            "query_embedding": query_embedding,
            "match_user_id": user_id,
            "match_count": top_k,
            "match_threshold": threshold,
        }
        result = client.rpc("match_documents", fallback_params).execute()
        return result.data or []


async def get_user_has_documents(user_id: str) -> bool:
    client = get_admin_client()
    result = (
        client.table("documents")
        .select("id", count="exact")
        .eq("user_id", user_id)
        .eq("status", "ready")
        .limit(1)
        .execute()
    )
    return (result.count or 0) > 0
