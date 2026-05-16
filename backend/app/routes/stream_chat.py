"""Streaming chat endpoint - SSE with two-phase message persistence."""

import re
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

        # Query structured data with smart dataset routing
        if structured_data:
            try:
                from app.llm.orchestrator import orchestrator
                ranked_datasets = _rank_datasets(message, structured_data)
                for dataset in ranked_datasets[:3]:
                    result = await query_structured_data(
                        message, dataset, orchestrator.llm, user_id=user_id
                    )
                    if result and not _is_unhelpful_result(result):
                        code_result = result
                        break
                if code_result is None and ranked_datasets:
                    code_result = result
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


def _rank_datasets(query: str, datasets: list[dict]) -> list[dict]:
    """Rank datasets by relevance to the user's query using keyword matching."""
    query_lower = query.lower()
    query_words = set(re.findall(r'[a-z]+', query_lower))

    scored = []
    for ds in datasets:
        score = 0
        filename_lower = ds.get("filename", "").lower()
        columns_lower = [c.lower() for c in ds.get("columns", [])]
        all_col_text = " ".join(columns_lower)

        for word in query_words:
            if word in filename_lower:
                score += 3
            for col in columns_lower:
                if word in col:
                    score += 2
                if word == col:
                    score += 3

        domain_signals = {
            "churn": ["churn", "retention", "attrition", "customer_left", "left"],
            "price": ["price", "pricing", "cost", "amount", "revenue", "demand"],
            "marketing": ["marketing", "campaign", "conversion", "click", "impression", "roi", "engagement"],
            "sentiment": ["sentiment", "review", "comment", "text", "opinion", "feedback", "polarity"],
            "sales": ["sales", "revenue", "profit", "order", "quantity", "discount"],
        }

        for domain, signals in domain_signals.items():
            if any(s in query_lower for s in signals):
                if any(s in filename_lower or s in all_col_text for s in signals):
                    score += 5

        scored.append((score, ds))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [ds for _, ds in scored]


def _is_unhelpful_result(result: str) -> bool:
    """Check if a code query result is unhelpful/empty."""
    if not result:
        return True
    unhelpful_patterns = [
        "does not contain", "not possible", "no data available",
        "not found", "cannot", "doesn't have", "no .* information",
        "code rejected", "query execution error", "no result",
        "empty", "no column",
    ]
    result_lower = result.lower()
    return any(re.search(p, result_lower) for p in unhelpful_patterns)


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
        "You are a business analytics assistant. You help users make smarter decisions "
        "about pricing, customer retention, marketing, and customer sentiment using their own data."
    )

    if model_status:
        trained = [name for name, info in model_status.items() if info.get("trained")]
        if trained:
            base += "\n\nYou have trained prediction models ready to answer what-if questions."

    if simulation_result:
        base += f"""

INTERNAL DATA (use to form your answer, but do NOT expose raw numbers or model names):
{simulation_result}

Translate these results into plain business language. State the expected outcome directly."""

    if rag_context:
        # Strip source metadata before injecting
        cleaned_context = re.sub(r'\[Source:.*?\|.*?\]\n?', '', rag_context).strip()
        base += f"""

BACKGROUND INFORMATION FROM USER'S DATA:
{cleaned_context}

Use this information to support your answer. Do NOT reference filenames or sources."""

    if code_result:
        if not (_is_unhelpful_result(code_result) and simulation_result):
            base += f"""

COMPUTED ANSWER FROM USER'S DATA:
{code_result}

Present this result clearly in plain language."""

    base += """

RESPONSE RULES (strictly follow these):
- Speak in simple business language a CEO would understand.
- NEVER mention model metrics (r2, accuracy, MAE, RMSE), training data sizes, number of rows/features, or internal file names.
- NEVER say "the model predicts" or "according to the model". Just state the result as a fact from their data.
- When referencing data, say "based on your data" — never cite filenames or source tags.
- Give direct answers with specific numbers (percentages, dollar amounts, counts).
- Keep answers concise: 2-4 sentences for simple questions, short paragraphs for complex ones.
- End with 1-2 practical follow-up questions the user might want to ask next.
- If you don't have enough data to answer, say so briefly and suggest what data would help."""

    return base
