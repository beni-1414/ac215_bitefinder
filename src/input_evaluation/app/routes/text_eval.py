from __future__ import annotations
import time
from pathlib import Path
from fastapi import APIRouter
from app.schemas import TextEvalRequest, TextEvalResponse
from app.services.vertex_llm import get_llm

router = APIRouter(prefix="/v1/evaluate", tags=["evaluate"])


@router.post("/text", response_model=TextEvalResponse)
async def evaluate_text(req: TextEvalRequest) -> TextEvalResponse:
    start = time.perf_counter()

    if req.first_call:
        content = req.user_text
    else:
        content = "\n".join(req.history + [req.user_text])

    path = Path(__file__).parent / "prompt_template.md"
    template = Path(path).read_text(encoding="utf-8")
    prompt = template.format(content=content)

    llm = get_llm()
    result = llm.evaluate_text({"prompt": prompt})

    latency_ms = int((time.perf_counter() - start) * 1000)
    return TextEvalResponse(
        complete=bool(result.get("complete", False)),
        improve_message=result.get("improve_message"),
        combined_text=(
            result.get("combined_text")
            if not req.first_call and req.return_combined_text
            else None
        ),
        high_danger=result.get("high_danger", False),
        latency_ms=latency_ms,
    )
