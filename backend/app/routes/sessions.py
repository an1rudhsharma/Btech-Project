"""Session management endpoints - CRUD for chat sessions and messages."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.auth.middleware import get_current_user
from app.db import sessions as sessions_db

router = APIRouter()


class CreateSessionRequest(BaseModel):
    title: Optional[str] = "New Chat"


class UpdateSessionRequest(BaseModel):
    title: Optional[str] = None
    has_uploaded_data: Optional[bool] = None


@router.get("/sessions")
async def list_sessions(user: dict = Depends(get_current_user)):
    """List all chat sessions for the current user."""
    return await sessions_db.list_sessions(user["id"])


@router.post("/sessions")
async def create_session(req: CreateSessionRequest, user: dict = Depends(get_current_user)):
    """Create a new chat session."""
    session = await sessions_db.create_session(user["id"], req.title or "New Chat")
    if not session:
        raise HTTPException(500, "Failed to create session")
    return session


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, user: dict = Depends(get_current_user)):
    """Get a specific session with its messages."""
    session = await sessions_db.get_session(user["id"], session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    messages = await sessions_db.get_messages(user["id"], session_id)
    return {"session": session, "messages": messages}


@router.patch("/sessions/{session_id}")
async def update_session(session_id: str, req: UpdateSessionRequest, user: dict = Depends(get_current_user)):
    """Update session metadata (title, flags)."""
    updates = {k: v for k, v in req.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(400, "No fields to update")
    result = await sessions_db.update_session(user["id"], session_id, updates)
    return result


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user: dict = Depends(get_current_user)):
    """Delete a chat session and all its messages."""
    await sessions_db.delete_session(user["id"], session_id)
    return {"status": "deleted"}


@router.get("/sessions/{session_id}/messages")
async def get_messages(session_id: str, user: dict = Depends(get_current_user)):
    """Get all messages for a session."""
    messages = await sessions_db.get_messages(user["id"], session_id)
    return messages
