"""Streaming chat endpoint - SSE with two-phase message persistence."""

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import json
import asyncio

from app.auth.middleware import get_current_user
from app.db import sessions as sessions_db
from app.rag.retriever import retrieve_context, retrieve_structured_data_info
from app.rag.code_query import query_structured_data
from app.config import settings

router = APIRouter()


class StreamChatRequest(BaseModel):
    message: str
    session_id: str


@router.post("/chat/stream")
async def stream_chat(req: StreamChatRequest, user: dict = Depends(get_current_user)):
    """
    Streaming chat with RAG context injection.
    Two-phase persistence: save user msg first, then stream + save assistant msg.
    """
    session = await sessions_db.get_session(user["id"], req.session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    # Phase 1: Save user message immediately
    await sessions_db.save_message(req.session_id, "user", req.message)

    # Update session title from first message
    if session.get("title") == "New Chat":
        title = req.message[:50].strip()
        await sessions_db.update_session(user["id"], req.session_id, {"title": title})

    # Phase 2: Generate and stream response
    return StreamingResponse(
        _generate_stream(user["id"], req.session_id, req.message),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _generate_stream(user_id: str, session_id: str, message: str):
    """Generator that yields SSE events."""
    full_response = ""

    try:
        # Retrieve RAG context
        rag_context = await retrieve_context(user_id, message)

        # Check for structured data queries
        structured_data = await retrieve_structured_data_info(user_id)
        code_result = None

        quantitative_keywords = ["total", "sum", "average", "mean", "count", "how many",
                                 "maximum", "minimum", "group by", "compare", "percentage"]
        is_quantitative = any(kw in message.lower() for kw in quantitative_keywords)

        if structured_data and is_quantitative:
            from app.llm.orchestrator import orchestrator
            code_result = await query_structured_data(
                message, structured_data[0], orchestrator.llm
            )

        # Build system prompt with context
        system_prompt = _build_system_prompt(rag_context, code_result)

        # Stream from Groq
        from groq import Groq
        client = Groq(api_key=settings.groq_api_key)

        # Get recent conversation history for context
        recent_messages = await sessions_db.get_messages(user_id, session_id, limit=10)
        messages = [{"role": "system", "content": system_prompt}]
        for msg in recent_messages[-8:]:  # Last 8 messages for context
            if msg["status"] == "complete":
                messages.append({"role": msg["role"], "content": msg["content"][:2000]})
        messages.append({"role": "user", "content": message})

        stream = client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            temperature=0.3,
            max_tokens=2000,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                full_response += delta.content
                yield f"data: {json.dumps({'token': delta.content})}\n\n"

        # Save complete assistant message
        await sessions_db.save_message(session_id, "assistant", full_response)
        yield f"data: {json.dumps({'done': True})}\n\n"

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        if not full_response:
            await sessions_db.save_message(session_id, "assistant", error_msg)
        yield f"data: {json.dumps({'error': error_msg})}\n\n"


def _build_system_prompt(rag_context: str, code_result: Optional[str]) -> str:
    """Build the system prompt with RAG context and code query results."""
    base = (
        "You are an AI business analytics assistant. You help users understand their data, "
        "run simulations, and make data-driven decisions about pricing, churn, marketing, and sentiment."
    )

    if rag_context:
        base += f"""

RELEVANT CONTEXT FROM USER'S DOCUMENTS:
{rag_context}

Use the above context to inform your responses. Cite the sources when relevant.
If the context doesn't contain relevant information for the question, say so and provide general guidance."""

    if code_result:
        base += f"""

DATA QUERY RESULT:
The following is a computed answer from the user's uploaded dataset:
{code_result}

Present this result clearly and provide any relevant analysis or interpretation."""

    base += """

Guidelines:
- Be concise and actionable
- When discussing data, cite specific numbers from the context
- If you don't have enough information, say so honestly
- Suggest follow-up questions the user might want to ask"""

    return base
