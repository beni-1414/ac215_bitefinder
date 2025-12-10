from __future__ import annotations
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from api.schemas import RAGRequest
from api.services.clients import post_rag_chat, ServiceError
from api.services.vertex_llm import get_llm
import inspect

router = APIRouter(prefix="/v1/orchestrator", tags=["rag"])


@router.post("/rag")
def orchestrator_rag(req: RAGRequest) -> Dict[str, Any]:
    try:
        rag_payload = post_rag_chat(req)
        # print("RAG PAYLOAD:", rag_payload)

    except ServiceError as e:
        raise HTTPException(status_code=502, detail=f"ragmodel service error: {str(e)}")

    # ragmodel returns: { status: "ok", payload: {...}, latency_ms: ... }
    if rag_payload.status != "ok":
        raise HTTPException(status_code=500, detail="ragmodel returned non-ok status")

    inner = rag_payload.payload
    prompt = inner.prompt or ""
    context = inner.context or ""

    # Fallback prompt if ragmodel returned nothing
    if not prompt.strip():
        prompt = (
            f"User question: '{req.question}'.\n"
            f"Detected insect: {req.bug_class or 'unknown'}.\n"
            f"Reported symptoms: {req.symptoms or ''}.\n"
            f"Confidence: {req.conf or 0.0}.\n\n"
            "RAG context is unavailable. Provide a helpful general answer "
            "based only on the information above."
        )

    try:
        llm = get_llm()
        print("VertexLLM dir:", dir(llm))
        print("rag file - VertexLLM loaded from:", inspect.getfile(llm.__class__))

        llm_resp = llm.evaluate_text({"prompt": prompt})

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Vertex LLM error: {str(e)}")

    return {
        "context": context,
        "llm": llm_resp,
    }
