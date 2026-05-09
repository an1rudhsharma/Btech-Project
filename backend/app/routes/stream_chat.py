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

        # Query structured data if datasets exist
        if structured_data:
            try:
                from app.llm.orchestrator import orchestrator
                code_result = await query_structured_data(
                    message, structured_data[0], orchestrator.llm, user_id=user_id
                )
            except Exception:
                pass

        # Check ML model status and run simulation if applicable
        from app.engine.simulation import get_simulation_engine
        sim_engine = get_simulation_engine()
        model_status = sim_engine.get_model_status()
        simulation_result = None

        simulation_keywords = ["what happens if", "what if", "raise", "lower", "increase",
                               "decrease", "change", "simulate", "impact", "effect",
                               "reduce", "double", "triple", "cut", "boost"]
        is_simulation = any(kw in message.lower() for kw in simulation_keywords)

        if is_simulation and any(v.get("trained") for v in model_status.values()):
            simulation_result = _try_simulation(message, sim_engine)

        # Build system prompt with context
        system_prompt = _build_system_prompt(rag_context, code_result, model_status, simulation_result)

        # Stream from Groq
        from groq import Groq
        client = Groq(api_key=settings.groq_api_key)

        # Get recent conversation history for context
        recent_messages = await sessions_db.get_messages(user_id, session_id, limit=10)
        messages = [{"role": "system", "content": system_prompt}]
        for msg in recent_messages[-8:]:
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


def _try_simulation(message: str, sim_engine) -> Optional[str]:
    """Attempt to extract scenario parameters and run a simulation."""
    import re

    scenario = {}
    msg_lower = message.lower()

    # Extract price changes
    price_match = re.search(r'price.*?(\d+)%', msg_lower) or re.search(r'(\d+)%.*?price', msg_lower)
    if price_match:
        pct = int(price_match.group(1))
        if "raise" in msg_lower or "increase" in msg_lower or "up" in msg_lower:
            scenario["price"] = 100 * (1 + pct / 100)
        elif "lower" in msg_lower or "reduce" in msg_lower or "cut" in msg_lower or "decrease" in msg_lower:
            scenario["price"] = 100 * (1 - pct / 100)
        else:
            scenario["price"] = 100 * (1 + pct / 100)

    # Extract marketing spend changes
    mktg_match = re.search(r'marketing.*?(\d+)%', msg_lower) or re.search(r'(\d+)%.*?marketing', msg_lower)
    if mktg_match:
        pct = int(mktg_match.group(1))
        if "double" in msg_lower:
            scenario["marketing_spend"] = 20000
        elif "reduce" in msg_lower or "cut" in msg_lower or "decrease" in msg_lower:
            scenario["marketing_spend"] = 10000 * (1 - pct / 100)
        else:
            scenario["marketing_spend"] = 10000 * (1 + pct / 100)

    if "double" in msg_lower and "marketing" in msg_lower:
        scenario["marketing_spend"] = 20000

    # Default scenario if nothing specific extracted
    if not scenario:
        scenario = {"price": 125, "marketing_spend": 10000}

    try:
        text = "product is good"
        if "bad" in msg_lower or "negative" in msg_lower:
            text = "product is terrible"
        result = sim_engine.simulate(scenario, text=text)
        if result.get("predictions"):
            parts = ["**Simulation Results:**"]
            for model_name, pred in result["predictions"].items():
                if isinstance(pred, dict):
                    for k, v in pred.items():
                        if isinstance(v, (int, float)):
                            parts.append(f"- {model_name}.{k}: {v:.4f}")
                elif isinstance(pred, (int, float)):
                    parts.append(f"- {model_name}: {pred:.4f}")
            return "\n".join(parts)
    except Exception:
        pass
    return None


def _build_system_prompt(rag_context: str, code_result: Optional[str],
                         model_status: Optional[dict] = None,
                         simulation_result: Optional[str] = None) -> str:
    """Build the system prompt with RAG context and code query results."""
    base = (
        "You are an AI business analytics assistant. You help users understand their data, "
        "run simulations, and make data-driven decisions about pricing, churn, marketing, and sentiment."
    )

    # Include ML model capabilities
    if model_status:
        trained = [name for name, info in model_status.items() if info.get("trained")]
        if trained:
            base += f"""

TRAINED ML MODELS AVAILABLE: {', '.join(trained)}
You can run simulations using these models. When users ask "what if" questions about pricing, churn, marketing, or sentiment, you have real ML predictions to back up your answers."""

    if simulation_result:
        base += f"""

SIMULATION RESULT (from trained ML models):
{simulation_result}

Interpret these simulation results for the user. Explain what the numbers mean in business terms."""

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
