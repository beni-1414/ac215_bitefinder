from typing import Any, Dict, Optional
import os
import uuid

from fastapi import APIRouter, HTTPException

from api.schemas import RAGRequest
from api.config import settings
from api.services.tools import get_rag_answer  # your existing RAG tool

from vertexai import agent_engines
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from langchain_google_firestore import FirestoreChatMessageHistory
from google.cloud import firestore

router = APIRouter(prefix="/v1/orchestrator", tags=["rag-agent"])

# -------------------------------------------------------------------
# System instruction: same spirit as your ADK version
# -------------------------------------------------------------------
AGENT_SYSTEM_INSTRUCTION = """
You are BiteFinder, a polite, concise bug-bite assistant.
- Handle greetings, thanks, or irrelevant questions courteously and redirect to bug-bite help.
- Decide when to call the tool get_rag_answer to fetch detailed, up-to-date bug-bite context.
- Prefer tool results when available; if not, answer based on your training data.
- Always answer in one short paragraph. No lists, no markdown, no JSON.
- Feel free to answer any follow-up questions as long as they relate to bug-bites.
- A question like "Where can I find this cream or product?" is relevant if it relates to bug-bite treatment or care.

IMPORTANT: Output ONLY a single, concise, evidence-based PARAGRAPH, summarizing the answer to the user's question.
Do NOT use any list, JSON, array, or field structure. Do not output markdown code blocks or any delimiters.
No bold section headings or asterisks. Never prefix or suffix your answer.
Never mention you are an AI, or apologize for missing data.
""".strip()


# -------------------------------------------------------------------
# Model + safety config
# -------------------------------------------------------------------

MODEL_NAME = settings.VERTEX_MODEL_NAME  # e.g. "gemini-2.0-flash"

# Optional safety settings (tweak as you like)
safety_settings = {
    HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}

model_kwargs = {
    "temperature": 0.28,
    "max_output_tokens": 512,
    "top_p": 0.95,
    "top_k": None,
    "safety_settings": safety_settings,
}


# -------------------------------------------------------------------
# Firestore-backed chat history
# -------------------------------------------------------------------

FIRESTORE_COLLECTION = settings.FIRESTORE_COLLECTION
FIRESTORE_DATABASE = settings.FIRESTORE_DATABASE

# 🔹 Make these lazy singletons instead of eager module-level objects
_firestore_client: Optional[firestore.Client] = None
_langchain_agent: Optional[agent_engines.LangchainAgent] = None


def get_firestore_client() -> firestore.Client:
    global _firestore_client

    if _firestore_client is None:
        project_id = os.getenv("GCP_PROJECT")
        if not project_id:
            raise RuntimeError("GCP_PROJECT must be set for Firestore.")
        _firestore_client = firestore.Client(
            project=project_id,
            database=FIRESTORE_DATABASE,
        )
    return _firestore_client


def get_session_history(session_id: str) -> FirestoreChatMessageHistory:
    """
    LangChain-compatible session history factory.

    Given a session_id, returns a FirestoreChatMessageHistory object that
    stores conversation turns for that session in Firestore.
    """
    return FirestoreChatMessageHistory(
        client=get_firestore_client(),
        session_id=session_id,
        collection=FIRESTORE_COLLECTION,
        encode_message=False,
    )


# -------------------------------------------------------------------
# LangChain Agent (Vertex AI LangchainAgent template)
# -------------------------------------------------------------------


def get_langchain_agent() -> agent_engines.LangchainAgent:
    global _langchain_agent

    if _langchain_agent is None:
        _langchain_agent = agent_engines.LangchainAgent(
            model=MODEL_NAME,
            system_instruction=AGENT_SYSTEM_INSTRUCTION,
            model_kwargs=model_kwargs,
            tools=[get_rag_answer],
            chat_history=get_session_history,
        )
    return _langchain_agent


def build_user_input(req: RAGRequest) -> str:
    """
    Flatten RAGRequest into a single user input string.

    The system_instruction already tells the agent how to behave,
    so this just passes structured info in a readable way.
    """
    return (
        f"User question: '{req.question}'.\n"
        f"Detected insect: {req.bug_class or 'unknown'}.\n"
        f"Reported symptoms: {req.symptoms or ''}.\n"
        # f"Model confidence in detection: {req.conf or 0.0}.\n"
        "Answer following the BiteFinder rules."
    )


@router.post("/rag-agent")
async def run_rag_agent(req: RAGRequest) -> Dict[str, Any]:
    """
    LangChain-based agent endpoint that can replace /rag:

    - Accepts the same payload shape as /rag.
    - Lets LangchainAgent decide whether/when to call get_rag_answer.
    - Uses Firestore to store per-session chat history.
    - Returns a compact answer paragraph (non-streaming).
    """
    if not req.question:
        raise HTTPException(status_code=400, detail="Question is required.")

    # Session ID is purely app-level: one per browser conversation.
    # Frontend should send an existing session_id if it has one; otherwise we create one.
    session_id: Optional[str] = getattr(req, "session_id", None)
    if not session_id:
        session_id = str(uuid.uuid4())

    user_input = build_user_input(req)

    try:
        # Pass session_id via config.configurable so LangchainAgent
        # knows which FirestoreChatMessageHistory instance to use.
        response = get_langchain_agent().query(
            input=user_input,
            config={"configurable": {"session_id": session_id}},
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Agent error: {e}")

    # LangchainAgent returns dict like {"input": "...", "output": "..."}
    answer = ""
    if isinstance(response, dict):
        answer = (response.get("output") or "").strip()
    else:
        answer = str(response).strip()

    if not answer:
        answer = "Sorry, I could not generate an answer right now."

    print("AGENT CALLED!!")

    return {
        "llm": answer,
        "session_id": session_id,
        # keeping user_id for future; you can hardcode "anonymous" or drop it
        "user_id": "anonymous",
    }
