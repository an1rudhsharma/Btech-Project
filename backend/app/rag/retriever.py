"""RAG retriever - searches pgvector for relevant document chunks."""

from app.rag.embeddings import embed_text
from app.db.knowledge import search_chunks, get_user_has_documents


async def retrieve_context(user_id: str, query: str, top_k: int = 8, document_ids: list[str] = None) -> str:
    """
    Retrieve relevant document chunks for a query.
    Returns formatted context string ready for injection into the system prompt.
    Returns empty string if user has no documents.
    
    Args:
        user_id: The user's ID
        query: Natural language query to search for
        top_k: Number of top chunks to retrieve
        document_ids: Optional list of document IDs to scope the search to
    """
    has_docs = await get_user_has_documents(user_id)
    if not has_docs:
        return ""

    query_embedding = embed_text(query)
    results = await search_chunks(
        user_id, query_embedding, top_k=top_k, threshold=0.15, document_ids=document_ids
    )

    if not results:
        return ""

    context_parts = []
    for i, chunk in enumerate(results, 1):
        similarity = chunk.get("similarity", 0)
        content = chunk.get("content", "")
        metadata = chunk.get("metadata", {})
        source = metadata.get("filename", "document")
        context_parts.append(f"[Source: {source} | Relevance: {similarity:.0%}]\n{content}")

    context = "\n\n---\n\n".join(context_parts)
    return context


async def retrieve_structured_data_info(user_id: str) -> list[dict]:
    """Get metadata about queryable (CSV/Excel) datasets the user has uploaded."""
    from app.db.supabase_client import get_admin_client
    client = get_admin_client()
    result = (
        client.table("documents")
        .select("id, filename, metadata")
        .eq("user_id", user_id)
        .eq("status", "ready")
        .execute()
    )
    queryable = []
    for doc in (result.data or []):
        meta = doc.get("metadata", {})
        if meta.get("queryable"):
            queryable.append({
                "doc_id": doc["id"],
                "filename": doc["filename"],
                "columns": meta.get("columns", []),
                "dtypes": meta.get("dtypes", {}),
                "shape": meta.get("shape", []),
                "sample_rows": meta.get("sample_rows", []),
            })
    return queryable
