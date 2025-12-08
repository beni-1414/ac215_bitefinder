# This file defines FastAPI routing logic, imports service layers and connects them to HTTP routes.
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv, find_dotenv

# import modified chat() and agent() functions
from api.services.rag_pipeline import chat

###
from api.services.pinecone_adapter import query_by_vector
from api.services.rag_pipeline import generate_query_embedding
###


load_dotenv(find_dotenv())

router = APIRouter()


# request schema
class ChatRequest(BaseModel):
    question: str | None = None
    symptoms: str = ""
    conf: float = 0.0
    bug_class: str = ""


@router.post("/chat")
def rag_preprocess_chat(request: ChatRequest):
    try:
        payload = chat(
            symptoms=request.symptoms or "",
            conf=request.conf,
            bug_class=request.bug_class,
            question=getattr(request, "question", ""),
        )

        # THIS WRAPPER IS REQUIRED
        return {"status": "ok", "payload": payload, "latency_ms": 0}

    except Exception as e:
        import traceback

        print("❌ RAG /chat error:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

###
# request models
'''
class SearchByBugRequest(BaseModel):
    bug: str
    search_content: str
    conf: float | None = 0.0
    question: str | None = None 

class SearchBySymptomRequest(BaseModel):
    search_content: str
    question: str | None = None

@router.post("/search_by_bug")
def rag_search_by_bug(req: SearchByBugRequest):
    try:
        query_text = (req.question or req.search_content or f"General info about {req.bug} bites").strip()
        qvec = generate_query_embedding(f"{query_text}. Bug type: {req.bug}")
        results = query_by_vector(
            query_vec=qvec,
            top_k=10,
            metadata_filter={"bug": {"$eq": req.bug}},
        )
        docs = results.get("documents", [[]])[0]
        context = "\n".join(docs) if docs else ""
        payload = {
            "question": query_text,
            "prompt": (
                f"User question: {query_text}\n"
                f"Bug: {req.bug}\n"
                f"Confidence: {req.conf or 0.0}\n\n"
                f"Context:\n{context}"
            ),
            "context": context,
            "bug_class": req.bug,
            "conf": req.conf or 0.0,
        }
        return {"status": "ok", "payload": payload, "latency_ms": 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search_by_symptom")
def rag_search_by_symptom(req: SearchBySymptomRequest):
    try:
        query_text = (req.question or req.search_content).strip()
        qvec = generate_query_embedding(query_text)
        results = query_by_vector(query_vec=qvec, top_k=10)
        docs = results.get("documents", [[]])[0]
        context = "\n".join(docs) if docs else ""
        payload = {
            "question": query_text,
            "prompt": (
                f"User question: {query_text}\n"
                f"Symptoms/content: {req.search_content}\n\n"
                f"Context:\n{context}"
            ),
            "context": context,
        }
        return {"status": "ok", "payload": payload, "latency_ms": 0}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        '''
###