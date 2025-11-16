# This file defines FastAPI routing logic, imports service layers and connects them to HTTP routes.
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import time
from dotenv import load_dotenv, find_dotenv

# import modified chat() and agent() functions
from api.routes.rag_pipeline import chat

load_dotenv(find_dotenv())

router = APIRouter()

# request schema
class ChatRequest(BaseModel):
    symptoms: str
    conf: float
    bug_class: str

@router.post("/chat")
def rag_preprocess_chat(request: ChatRequest):
    start_time = time.time()
    try:
        # run pre-LLM pipeline (embedding, retrieval, prompt assembly)
        payload = chat(
            symptoms=request.symptoms,
            conf=request.conf,
            bug_class=request.bug_class
        )

        latency_ms = int((time.time() - start_time) * 1000)
        return JSONResponse({
            "status": "ok",
            "payload": payload,
            "latency_ms": latency_ms
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))