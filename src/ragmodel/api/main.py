# FastAPI entrypoint for the RAG preprocessor service
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from api.routes import rag_router 

api = FastAPI(title="RAG Preprocessor Service")

api.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ include the router with the /rag prefix
api.include_router(rag_router.router, prefix="/rag")

@api.get("/")
def root():
    return {"status": "running"}
