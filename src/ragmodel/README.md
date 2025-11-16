# RAG Model Service

This service implements a modular **retrieval-augmented generation (RAG)** pipeline for the BiteFinder project.  
It preprocesses user input (symptoms, confidence score, bug class), generates embeddings, retrieves context from Pinecone, and builds a structured prompt.  
Instead of directly calling a language model, it **stores the prepared payload in a shared container**, allowing a separate orchestrator API to handle the actual LLM response generation.

---

## Repository Overview

### Core Components

| File                            | Description                                                                                                                                                |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `api/routes/rag_router.py`          | Defines FastAPI route (`/rag/chat`) and connects incoming API requests to the underlying RAG pipeline.                                |
| `api/services/rag_pipeline.py`        | Implements the core RAG workflow: chunking, embedding generation (Vertex AI), vector retrieval (Pinecone), prompt assembly, and response payload creation. |
| `api/services/agent_tools.py`       | Provides Gemini / Vertex AI tool-calling utilities and helper methods for structured responses.                                                            |
| `api/services/pinecone_adapter.py`  | Manages upserting, querying, and retrieval of embeddings in the Pinecone vector database.                                                                  |
| `api/services/semantic_splitter.py` | Performs text chunking and semantic splitting when building or updating the embedding database.                                                            |
| `api/scripts/create_index.py`       | One-time utility script for creating and initializing the Pinecone index. Intended for manual setup, not runtime execution.                                |
| `api/main.py`       | FastAPI application entrypoint - starts the RAG Preprocessor API server, loads routes, and exposes HTTP endpoint `/rag/chat` |
| `api/__init__.py`               | Marks the `api` directory as a Python package and exposes submodules for routes, services, and scripts.                                                    |
| `Dockerfile`                    | Defines the build instructions for containerizing the RAG application.                                                                                     |
| `docker-compose.yml`            | Configures how the RAG container runs and interacts with other services (e.g., orchestrator, databases).                                                   |
| `pyproject.toml`                | Declares your project’s metadata, dependencies, and package structure. Ensures that `api/` is treated as the installable Python package for imports like `api.services.*`. |


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

2. **RAG Preprocessing (inside `api/services/rag_pipeline.py`)**
- Generates an embedding for the user query using Vertex AI’s text-embedding-004 model.
- Queries Pinecone to retrieve the most relevant text chunks for the specified bug class.
- Builds a structured prompt that combines:
    - the user’s symptoms
    - the model’s predicted bug type
    - retrieved context from the vector database
- Returns this assembled payload directly to the calling service.
- Note: The RAG Preprocessor does not call Gemini. It only prepares the structured prompt and context (see below)

3. **LLM Orchestration (external)**
- Another service (your orchestrator API or Gemini agent) is responsible for:
    - Calling /rag/chat
    - Receiving the returned payload (question, prompt, context)
    - Making the actual LLM call to Gemini
    - Returning the LLM’s final answer to the user or frontend

## Initializing the Vector Database (required before first use)

To use the RAG Preprocessor, you must first build and populate the Pinecone vector database from inside `/ragmodel`:
```
python api/scripts/build_vector_store.py
```
This pipeline:
- splits bug-bite text files into chunks
- generates embeddings with Vertex AI
- upserts them into Pinecone
- Once this is done, vector DB is ready for retrieval.

## Run the API Locally
Run from inside the `ragmodel/` directory:
```
uvicorn api.main:api --reload --port 8080
```

## Example Test: 
```
curl -X POST http://127.0.0.1:8080/rag/chat \
     -H "Content-Type: application/json" \
     -d '{"symptoms": "itchy red bumps on my arm", "conf": 0.9, "bug_class": "mosquito"}'
```