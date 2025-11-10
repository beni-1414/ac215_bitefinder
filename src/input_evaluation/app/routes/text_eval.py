from __future__ import annotations
import time
from fastapi import APIRouter
from app.schemas import TextEvalRequest, TextEvalResponse
from app.config import settings
from app.services.vertex_llm import get_llm

router = APIRouter(prefix="/v1/evaluate", tags=["evaluate"])

@router.post("/text", response_model=TextEvalResponse)
async def evaluate_text(req: TextEvalRequest) -> TextEvalResponse:
    start = time.perf_counter()

    if req.first_call:
        content = req.user_text
    else:
        content = "\n".join(req.history + [req.user_text])

    # You will craft the prompt. Placeholder template below.
    prompt = (
        "You are a form completeness checker for a bug bite identification app. Given the following user-provided description, "
        "respond ONLY with JSON: {\"complete\": <bool>, \"improve_message\": <string or null>, \"combined_text\": <string or null>, \"high_danger\": <bool>}\n\n"
        f"CONTENT:\n{content}\n\n"
        "Rules:\n"
        "The text must include **symptoms** and some **location information** regarding where the bite occurred (for example: in the park, at home...). Any information about the appearance of the bite, the body location, timing of the bite is useful but not strictly required.\n"
        "\n"
        "If details are missing, set complete=false and list missing items in improve_message in a friendly manner. If all required details are present, set complete=true and improve_message=null.\n"
        "If first_call=false and history is not empty, produce a concise combined_text merging all chunks; otherwise combined_text=null.\n"
        "If there is any mention of high danger symptoms (e.g., trouble breathing, face swelling), always set complete=false, high_danger=true, and improve_message=\"Your description indicates potential high danger symptoms. Please seek immediate medical attention.\"\n"
        "Considerations:\n"
        "- Be lenient with minor typos or informal language. Only ask for improvements if the information is clearly missing.\n"
        "- Keep improve_message concise and user-friendly, specifying what is missing. Example: 'To ensure proper classification, can you include where you think the bite occurred (home, park, etc.)?' Do not just answer 'Please provide more detail'\n"
        "- Ensure combined_text is well-structured and free of redundancy.\n"
        "- Do not raise high danger flag lightly; only for serious symptoms.\n"

    )

    llm = get_llm()
    result = llm.evaluate_text({"prompt": prompt})

    latency_ms = int((time.perf_counter() - start) * 1000)
    return TextEvalResponse(
        complete=bool(result.get("complete", False)),
        improve_message=result.get("improve_message"),
        combined_text=(result.get("combined_text") if not req.first_call and req.return_combined_text else None),
        high_danger=result.get("high_danger", False),
        latency_ms=latency_ms,
    )