from __future__ import annotations
import time
from pathlib import Path
from fastapi import APIRouter
from api.schemas import TextEvalRequest, TextEvalResponse
from api.services.vertex_llm import get_llm
import inspect

router = APIRouter(prefix="/v1/evaluate", tags=["evaluate"])


@router.post("/text", response_model=TextEvalResponse)
async def evaluate_text(req: TextEvalRequest) -> TextEvalResponse:
    start = time.perf_counter()
    llm = get_llm()

    # ----------------------------------------------------
    # FIRST MESSAGE — CHECK COMPLETENESS + RELEVANCE
    # ----------------------------------------------------
    if req.first_call:
        template_path = Path(__file__).parent / "prompt_template_initial.md"  # ← FIX THIS
        template = template_path.read_text(encoding="utf-8")
        prompt = template.replace("{content}", req.user_text)  # ← Note: {content} not {user_message}

        print(f"DEBUG INITIAL: user_text = {req.user_text!r}")
        print(f"DEBUG INITIAL: prompt preview = {prompt[:500]}")

        result = llm.evaluate_text({"prompt": prompt})
        print("text_eval file - VertexLLM loaded from:", inspect.getfile(llm.__class__))

        latency_ms = int((time.perf_counter() - start) * 1000)

        return TextEvalResponse(
            complete=bool(result.get("complete", False)),
            improve_message=result.get("improve_message"),
            combined_text=None,
            high_danger=result.get("high_danger", False),
            question_relevant=result.get("question_relevant", False),
            courtesy=result.get("courtesy", False),
            latency_ms=latency_ms,
        )

    # ----------------------------------------------------
    # FOLLOW-UP MESSAGE — CHECK RELEVANCE + COURTESY
    # ----------------------------------------------------
    template_path = Path(__file__).parent / "prompt_template_followup.md"
    template = template_path.read_text(encoding="utf-8")
    prompt = template.replace("{user_message}", req.user_text)

    print(f"DEBUG FOLLOWUP: user_text = {req.user_text!r}")  # ADD THIS
    print(f"DEBUG FOLLOWUP: prompt preview = {prompt[:500]}")  # ADD THIS

    result = llm.evaluate_text({"prompt": prompt})

    print(f"DEBUG FOLLOWUP: result = {result}")

    # Handle missing fields from LLM
    question_relevant = result.get("question_relevant")
    if question_relevant is None:
        # LLM didn't return the field - default to False
        question_relevant = False

    courtesy = result.get("courtesy")
    if courtesy is None:
        courtesy = False

    latency_ms = int((time.perf_counter() - start) * 1000)

    return TextEvalResponse(
        complete=True,
        improve_message=result.get("improve_message"),
        combined_text=None,
        high_danger=False,
        question_relevant=question_relevant,
        courtesy=courtesy,
        latency_ms=latency_ms,
    )
