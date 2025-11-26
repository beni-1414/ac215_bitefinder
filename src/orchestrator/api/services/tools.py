from typing import Optional

from api.schemas import RAGRequest
from api.services.clients import post_rag_chat, ServiceError


def get_rag_answer(
    question: str,
    symptoms: Optional[str] = None,
    bug_class: Optional[str] = None,
    conf: Optional[float] = None,
) -> str:
    """
    Tool callable by the ADK agent.
    Fetches prompt/context from the RAG service and returns a text block the model can use.
    """
    try:
        rag_resp = post_rag_chat(
            RAGRequest(
                question=question,
                symptoms=symptoms,
                bug_class=bug_class,
                conf=conf,
            )
        )
    except ServiceError as e:
        return f"[RAG unavailable: {e}]"
    except Exception as e:  # network resolution errors, etc.
        return f"[RAG call failed: {e}]"

    if rag_resp.status != "ok":
        return "[RAG returned non-ok status]"

    payload = rag_resp.payload
    context = payload.context or ""
    prompt = payload.prompt or ""

    print("RAG CALLED!!")

    # Provide both context and suggested prompt so the model can ground its answer.
    return "RAG_TOOL_OUTPUT:\n" f"Context:\n{context}\n\n" f"Prompt:\n{prompt}\n" "Use this information to answer."
