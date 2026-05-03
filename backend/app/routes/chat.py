"""Chat endpoint - natural language interface to the simulation system."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ChatMessage(BaseModel):
    message: str


class ReportRequest(BaseModel):
    scenarios: list[dict]


@router.post("/chat")
async def chat(msg: ChatMessage):
    """Process a natural language query through the full pipeline.

    Pipeline: Parse intent -> Simulate -> SHAP -> DiCE -> CoT Insight
    """
    from app.llm.orchestrator import orchestrator
    result = await orchestrator.process_natural_language_query(msg.message)
    return result


@router.post("/chat/report")
async def generate_report(request: ReportRequest):
    """Generate a full business report comparing multiple scenarios."""
    from app.llm.orchestrator import orchestrator
    report = await orchestrator.generate_business_report(request.scenarios)
    return {"report": report}
