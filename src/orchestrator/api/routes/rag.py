from __future__ import annotations
from typing import Any, Dict
from api.schemas import RAGRequest

from fastapi import APIRouter, HTTPException

from api.services.clients import post_rag_chat, ServiceError
from api.services.vertex_llm import get_llm
import pathlib

router = APIRouter(prefix="/v1/orchestrator", tags=["rag"])


def _load_prompt_template() -> str:
    p = pathlib.Path(__file__).parent / "prompt_template.md"
    return p.read_text(encoding="utf-8")


@router.post("/rag")
def orchestrator_rag(req: RAGRequest) -> Dict[str, Any]:
    """Call the rag model to retrieve context, then format a prompt including
    the user's question, symptoms, and predicted bite and call a Vertex LLM wrapper.

    Expected body keys: question, symptoms, conf (0..1), bug_class
    """
    question = req.question
    symptoms = req.symptoms or ""
    bug_class = req.bug_class or ""

    try:
        rag_resp = post_rag_chat(req)
    except ServiceError as e:
        raise HTTPException(status_code=502, detail=str(e))

    # Only the 'context' from RAG is relevant for building the LLM prompt
    if isinstance(rag_resp, dict):
        context = rag_resp.get("context", "")
    else:
        # RAGResponse Pydantic model: access attribute
        context = getattr(rag_resp, "context", "")

    # Build the prompt using the template and injected variables
    template = _load_prompt_template()
    prompt = template.format(
        context=context or "",
        question=question or "",
        symptoms=symptoms or "",
        bug_class=bug_class or "",
    )

    # Call the LLM
    try:
        llm = get_llm()
        llm_resp = llm.evaluate_text({"prompt": prompt})
    except Exception as e:
        # If Vertex/LLM fails, surface 502
        raise HTTPException(status_code=502, detail=str(e))

    return {"context": context, "llm": llm_resp}
