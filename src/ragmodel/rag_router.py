from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import time
from dotenv import load_dotenv, find_dotenv

# import modified chat() and agent() functions
from ragmodel.rag_pipeline import chat, agent

load_dotenv(find_dotenv())

router = APIRouter()
container = {}

# request schema
class ChatRequest(BaseModel):
    symptoms: str
    conf: float
    bug_class: str

@router.post("/chat")
def rag_preprocess_chat(request: ChatRequest):
    start_time = time.time()
    try:
        # run your pre-LLM pipeline (embedding, retrieval, prompt assembly)
        payload = chat(
            symptoms=request.symptoms,
            conf=request.conf,
            bug_class=request.bug_class
        )

        container["pending_llm_input"] = payload
        latency_ms = int((time.time() - start_time) * 1000)

        return JSONResponse({
            "status": "stored",
            "payload": payload,
            "latency_ms": latency_ms
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pending_llm_input")
def get_pending_llm_input():
    if "pending_llm_input" not in container:
        return {"status": "empty"}
    return container["pending_llm_input"]

# optional second endpoint to expose agent()
@router.post("/agent")
def rag_preprocess_agent():
    try:
        payload = agent()
        container["pending_llm_input"] = payload
        return {"status": "stored", "payload": payload}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
