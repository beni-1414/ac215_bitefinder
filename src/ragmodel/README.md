# RAG Model Service

This service implements a modular **retrieval-augmented generation (RAG)** pipeline for the BiteFinder project.  
It preprocesses user input (symptoms, confidence score, bug class), generates embeddings, retrieves context from Pinecone, and builds a structured prompt.  
Instead of directly calling a language model, it **stores the prepared payload in a shared container**, allowing a separate orchestrator API to handle the actual LLM response generation.

---

## Repository Overview

### Core Components

| File | Description |
|------|--------------|
| `rag_service.py` | FastAPI application entry point. Initializes the app, loads routers, and runs the API. |
| `rag_router.py` | Defines API endpoints (`/rag/chat`, `/pending_llm_input`) and routes requests to processing functions in `rag_pipeline.py`. |
| `rag_pipeline.py` | Handles the full RAG preprocessing pipeline: embedding generation (Vertex AI), vector retrieval (Pinecone), prompt construction, and payload storage inside a shared container. |
| `agent_tools.py` | Defines Gemini / Vertex AI tool and function-calling utilities. |
| `pinecone_adapter.py` | Contains helper functions for upserting and querying embeddings in Pinecone. |
| `semantic_splitter.py` | Handles text chunking and semantic splitting when constructing the vector database. |
| `create_index.py` | One-time script to initialize and populate the Pinecone index. |
| `__init__.py` | Marks `ragmodel` as a Python package (required for imports to work). |
| `Dockerfile` | Defines the container environment for deploying the RAG service. |
| `docker-shell.sh` | Opens an interactive shell inside the Docker container for debugging. |
| `docker-compose.yml` | Defines how this container runs and connects to other services (like the orchestrator). |

---

## How It Works

1. **User Input**

   The API receives a JSON body:
   ```json
   {
     "symptoms": "itchy red bumps on my arm",
     "conf": 0.9,
     "bug_class": "mosquito"
   }
   ```

2. **RAG Preprocessing (inside `rag_pipeline.py`)**
- Generates an embedding for the user query using Vertex AI’s text-embedding-004 model.
- Retrieves the most relevant context from Pinecone.
- Builds a structured prompt that combines retrieved text and metadata.
- Stores that payload in a shared in-memory container (container["pending_llm_input"]).

3. LLM Orchestration (external)
- The orchestrator API or Gemini agent fetches the stored payload from /rag/pending_llm_input.
- It then performs the actual LLM call and response generation.

## Run the API Locally
Run from inside the `src/` directory:
```
uvicorn ragmodel.rag_service:app --reload --port 8080
```

## Example Test: 
```
curl -X POST http://127.0.0.1:8080/rag/chat \
     -H "Content-Type: application/json" \
     -d '{"symptoms": "itchy red bumps on my arm", "conf": 0.9, "bug_class": "mosquito"}'
```