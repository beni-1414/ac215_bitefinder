# This file defines FastAPI routing logic, imports service layers and connects them to HTTP routes.
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import time
from dotenv import load_dotenv, find_dotenv

# import modified chat() and agent() functions
from api.services.rag_pipeline import chat

load_dotenv(find_dotenv())

router = APIRouter()


# request schema
class ChatRequest(BaseModel):
    question: str | None = None
    symptoms: str
    conf: float
    bug_class: str


@router.post("/chat")
def rag_preprocess_chat(request: ChatRequest):
    start_time = time.time()
    try:
        payload = chat(
            symptoms=request.symptoms,
            conf=request.conf,
            bug_class=request.bug_class,
            question=getattr(request, "question", ""),
        )
        latency_ms = int((time.time() - start_time) * 1000)
        return JSONResponse({"status": "ok", "payload": payload, "latency_ms": latency_ms})

    except Exception as e:
        import traceback

        print("❌ RAG /chat error:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
