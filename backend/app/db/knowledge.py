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
        .order("uploaded_at", desc=True)
        .execute()
    )
    return result.data or []


async def delete_document(user_id: str, doc_id: str) -> bool:
    client = get_admin_client()
    client.table("documents").delete().eq("id", doc_id).eq("user_id", user_id).execute()
    return True


async def insert_chunks(chunks: list[dict]) -> int:
    """Batch insert document chunks with embeddings."""
    if not chunks:
        return 0
    client = get_admin_client()
    client.table("document_chunks").insert(chunks).execute()
    return len(chunks)


async def search_chunks(user_id: str, query_embedding: list[float], top_k: int = 5, threshold: float = 0.3) -> list:
    """Vector similarity search using the match_documents RPC function."""
    client = get_admin_client()
    result = client.rpc("match_documents", {
        "query_embedding": query_embedding,
        "match_user_id": user_id,
        "match_count": top_k,
        "match_threshold": threshold,
    }).execute()
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
