"""Chat sessions and messages persistence layer."""

from typing import Optional
from app.db.supabase_client import get_admin_client


async def create_session(user_id: str, title: str = "New Chat") -> dict:
    client = get_admin_client()
    result = client.table("chat_sessions").insert({
        "user_id": user_id,
        "title": title,
    }).execute()
    return result.data[0] if result.data else {}


async def list_sessions(user_id: str) -> list:
    client = get_admin_client()
    result = (
        client.table("chat_sessions")
        .select("id, title, has_uploaded_data, created_at, updated_at")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .limit(50)
        .execute()
    )
    return result.data or []


async def get_session(user_id: str, session_id: str) -> Optional[dict]:
    client = get_admin_client()
    result = (
        client.table("chat_sessions")
        .select("*")
        .eq("id", session_id)
        .eq("user_id", user_id)
        .execute()
    )
    return result.data[0] if result.data else None


async def update_session(user_id: str, session_id: str, updates: dict) -> dict:
    client = get_admin_client()
    result = (
        client.table("chat_sessions")
        .update(updates)
        .eq("id", session_id)
        .eq("user_id", user_id)
        .execute()
    )
    return result.data[0] if result.data else {}


async def delete_session(user_id: str, session_id: str) -> bool:
    client = get_admin_client()
    client.table("chat_sessions").delete().eq("id", session_id).eq("user_id", user_id).execute()
    return True


async def get_messages(user_id: str, session_id: str, limit: int = 100) -> list:
    session = await get_session(user_id, session_id)
    if not session:
        return []
    client = get_admin_client()
    result = (
        client.table("messages")
        .select("id, role, content, status, created_at")
        .eq("session_id", session_id)
        .order("created_at", desc=False)
        .limit(limit)
        .execute()
    )
    return result.data or []


async def save_message(session_id: str, role: str, content: str, status: str = "complete") -> dict:
    client = get_admin_client()
    result = client.table("messages").insert({
        "session_id": session_id,
        "role": role,
        "content": content,
        "status": status,
    }).execute()
    # Touch the session's updated_at
    client.table("chat_sessions").update({"updated_at": "now()"}).eq("id", session_id).execute()
    return result.data[0] if result.data else {}


async def update_message(message_id: str, content: str, status: str = "complete") -> dict:
    client = get_admin_client()
    result = (
        client.table("messages")
        .update({"content": content, "status": status})
        .eq("id", message_id)
        .execute()
    )
    return result.data[0] if result.data else {}
