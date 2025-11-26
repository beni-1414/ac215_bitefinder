from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from api.schemas import RAGRequest

# Agent + tool wiring
from google.adk.agents import Agent  # ADK Agent
from vertexai.agent_engines import AdkApp

from api.config import settings
from api.services.tools import get_rag_answer  # RAG tool

router = APIRouter(prefix="/v1/orchestrator", tags=["rag-agent"])

AGENT_SYSTEM_INSTRUCTION = """
You are BiteFinder, a polite, concise bug-bite assistant.
- Handle greetings, thanks, or irrelevant questions courteously and redirect to bug-bite help.
- Decide when to call the tool get_rag_answer to fetch detailed, up-to-date bug-bite context.
- Prefer tool results when available; if not answer based on your training data.
- Always answer in one short paragraph. No lists, no markdown, no JSON.
- Feel free to answer any follow-up questions as long as they relate to bug-bites.
- A question like "Where can I find this cream or product?" is rellevant if it relates to bug-bite treatment or care.

IMPORTANT: Output ONLY a single, concise, evidence-based PARAGRAPH, summarizing the answer to the user's question.
Do NOT use any list, JSON, array, or field structure. Do not output markdown code blocks or any delimiters.
No bold section headings or asterisks. Never prefix or suffix your answer.
Never mention missing context—just answer based on what you know.
Never mention you are an AI, or apologize for missing data.
""".strip()

# Define the ADK Agent with the RAG tool
bitefinder_agent = Agent(
    model=settings.VERTEX_MODEL_NAME, name="bitefinder_rag_agent", instruction=AGENT_SYSTEM_INSTRUCTION, tools=[get_rag_answer]
)

# Wrap it in an AdkApp – this is what you query
bitefinder_app = AdkApp(agent=bitefinder_agent)


def build_user_message(req: RAGRequest) -> str:
    """Flatten the structured request into a single message for the agent."""
    return (
        f"User question: '{req.question}'.\n"
        f"Detected insect: {req.bug_class or 'unknown'}.\n"
        f"Reported symptoms: {req.symptoms or ''}.\n"
        f"Confidence: {req.conf or 0.0}.\n"
        "Choose whether to call get_rag_answer; then reply following the BiteFinder rules."
    )


def _extract_text(content: Any) -> str:
    """Extract concatenated text from an ADK/Vertex content payload."""
    if not content:
        return ""
    parts = []
    if isinstance(content, dict):
        parts = content.get("parts") or []
    elif hasattr(content, "parts"):
        parts = getattr(content, "parts") or []

    texts: list[str] = []
    for part in parts:
        # Part may be a dict or a genai Part with .text
        if isinstance(part, dict):
            txt = part.get("text")
            if txt:
                texts.append(txt)
        elif hasattr(part, "text"):
            txt = getattr(part, "text")
            if txt:
                texts.append(txt)
    return " ".join(texts).strip()


@router.post("/rag-agent")
async def run_rag_agent(req: RAGRequest) -> Dict[str, Any]:
    """
    ADK agent endpoint that can replace /rag:
    - Accepts the same payload shape as /rag.
    - Lets the agent decide whether/when to call the RAG tool.
    - Returns a compact answer paragraph (non-streaming; tool calls handled by ADK).
    """
    if not req.question:
        raise HTTPException(status_code=400, detail="Question is required.")

    # Derive user_id/session_id if you want multi-turn; stubbed for now.
    user_id: str = getattr(req, "user_id", None) or "anonymous"
    session_id: Optional[str] = getattr(req, "session_id", None)

    # If no session_id, create a new session for this user.
    if not session_id:
        try:
            session = await bitefinder_app.async_create_session(user_id=user_id)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Failed to create session: {e}")
        session_id = session.get("id") if isinstance(session, dict) else session.id

    # Stream events; keep the last text chunk (post tool-calls).
    final_answer = ""
    try:
        async for event in bitefinder_app.async_stream_query(
            user_id=user_id,
            session_id=session_id,
            message=build_user_message(req),
        ):
            content = event.get("content") if isinstance(event, dict) else getattr(event, "content", None)
            text = _extract_text(content)
            if text:
                final_answer = text
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Agent error: {e}")

    if not final_answer:
        final_answer = "Sorry, I could not generate an answer right now."

    return {
        "answer": final_answer.strip(),
        "session_id": session_id,
        "user_id": user_id,
    }
