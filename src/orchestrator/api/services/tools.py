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
    Tool for retrieving bug-bite specific, retrieval-augmented advice.

    Args:
        question: The user's natural-language question about a bug bite.
        symptoms: Optional free-text description of symptoms.
        bug_class: Optional detected insect class label (e.g. 'mosquito', 'tick').
        conf: Optional model confidence score for bug_class (0.0–1.0).

    Returns:
        A text block containing RAG context and a suggested prompt that the
        model should use to ground its final answer. If the RAG backend fails,
        returns a bracketed error message string instead.
    """
    print("RAG TOOL CALLED!!")
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
    except Exception as e:
        return f"[RAG call failed: {e}]"

    if rag_resp.status != "ok":
        return "[RAG returned non-ok status]"

    payload = rag_resp.payload
    context = payload.context or ""
    prompt = payload.prompt or ""

    print("RAG CALLED!!")

    return "RAG_TOOL_OUTPUT:\n" f"Context:\n{context}\n\n" f"Prompt:\n{prompt}\n" "Use this information to answer."
