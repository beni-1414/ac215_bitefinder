from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from ragmodel import rag_router  # ✅ use the full package path

app = FastAPI(title="RAG Preprocessor Service")

app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ include the router with the /rag prefix
app.include_router(rag_router.router, prefix="/rag")

@app.get("/")
def root():
    return {"status": "running"}
