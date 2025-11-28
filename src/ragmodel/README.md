# RAG Model Service

This service implements a modular retrieval-augmented generation (RAG) pipeline for the BiteFinder project. It preprocesses user input (symptoms, confidence score, bug class), generates embeddings, retrieves context from Pinecone, and builds a structured prompt. Instead of directly calling a language model, it **returns a prepared payload** so that a separate orchestrator service can perform the actual LLM call.

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
| `api/scripts/build_vector_store.py` | Full vector-store builder. Recreates the Pinecone index (via `create_index.py`), then chunks text, generates embeddings, and upserts everything into Pinecone. Run this whenever you update the bug-bite dataset or want to completely rebuild embeddings. |
| `api/scripts/create_index.py`       | One-time utility that creates (or recreates) the Pinecone index: sets name, dimensions, and serverless config. Called automatically by `build_vector_store.py`, but can also be run manually if needed. |
| `api/main.py`       | FastAPI application entrypoint - starts the RAG Preprocessor API server, loads routes, and exposes HTTP endpoint `/rag/chat` |
| `api/__init__.py`               | Marks the `api` directory as a Python package and exposes submodules for routes, services, and scripts.                                                    |
| `tests/unit/test_rag_pipeline.py`         | Unit tests for the core RAG preprocessing logic implemented in `chat()`, using mocks for embeddings and vector search.               |
| `tests/unit/test_rag_router.py`           | Unit tests for the `/rag/chat` API route, verifying request/response structure and router–pipeline wiring.                           |
| `tests/integration/test_rag_chat_integration.py` | Integration tests for the `/rag/chat` endpoint using FastAPI’s `TestClient`, validating end-to-end wiring while mocking external cloud services.        |
| `Dockerfile`                    | Defines the build instructions for containerizing the RAG application.                                                                                     |
| `docker-compose.yml`            | Configures how the RAG container runs and interacts with other services (e.g., orchestrator, databases).                                                   |
| `docker-entrypoint.sh` | Entrypoint script for the RAG container. Activates the virtual environment inside the image, loads environment variables, and launches the FastAPI server (DEV mode drops you into a shell; PROD mode auto-starts `uvicorn`). |
| `docker-shell.sh` | Local development runner for the RAG service. Builds the Docker image, mounts the api/ code and secrets/ directory, exports env variables from env.dev, and launches the container in DEV mode on the shared bitefinder-network. |
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
- creates (or recreates) the Pinecone index using `create_index.py`
- splits bug-bite text files into chunks
- generates embeddings with Vertex AI
- upserts them into Pinecone
- Once this is done, vector DB is ready for retrieval.

## Run the RAG Service in Docker

The RAG service can also be launched as a container using the included shell wrapper.
This automatically builds the Docker image, mounts your code and secrets, and starts the service in development mode.

### Start the container
From inside src/ragmodel/, run:
```
sh docker-shell.sh
```

### Inside the container (DEV mode)
Once the container starts, run:
```
uvicorn_server
```
The RAG API will be served at:
```http://localhost:9000/rag/chat```

### Test with curl
```
curl -X POST http://127.0.0.1:9000/rag/chat \
  -H "Content-Type: application/json" \
  -d '{"symptoms": "itchy red bumps", "conf": 0.9, "bug_class": "mosquito"}'
```

## Run the API Locally
Run from inside the `ragmodel/` directory:
```
uvicorn api.main:api --reload --port 8080
```

### Test with curl
```
curl -X POST http://127.0.0.1:8080/rag/chat \
     -H "Content-Type: application/json" \
     -d '{"symptoms": "itchy red bumps on my arm", "conf": 0.9, "bug_class": "mosquito"}'
```

## Testing
Tests live in: `ragmodel/tests/`. All tests must be run from inside the `ragmodel/` directory, since that is where the `api/` package root begins.

### Unit Tests
Unit tests verify core RAG preprocessing logic without calling external services (Vertex AI or Pinecone). Mocks simulate embeddings and vector search responses so the tests run fast and offline.
#### 1. `test_rag_pipeline.py`
**Validates the core RAG preprocessing logic implemented in chat().**

This tests ensures:
- embeddings are mocked correctly
- Pinecone retrieval is mocked
- the assembled payload includes
    - question
    - prompt
    - context
    - bug_class
- the context and prompt correctly incorporate both symptoms and retrieved chunks

**Run it with:**
```
python -m tests.unit.test_rag_pipeline
```

#### 2. `test_rag_router.py`
**Validates the /rag/chat API route using FastAPI’s TestClient.**

This test ensures:
- the route accepts a valid JSON request body
- the router calls the underlying chat() pipeline function (mocked)
- the response includes the correct structure:
    - "status": "ok"
    - "payload": {...}
    - "latency_ms" as an integer

**Run it with:**
```
python -m tests.unit.test_rag_router
```
### Integration Tests
Integration tests exercise the full FastAPI stack for the RAG service while still mocking external cloud dependencies (Vertex AI, Pinecone, Secret Manager), ensuring the HTTP layer and RAG wiring behave as expected.

#### 3. `tests/integration/test_rag_chat_integration.py`

**Validates the `/rag/chat` endpoint together with the RAG router using FastAPI’s `TestClient`.**

This test ensures:

- a realistic JSON request body with `symptoms`, `conf`, and `bug_class` is accepted by the API
- the router calls the underlying `chat()` pipeline function imported in `api.routes.rag_router` (mocked in the test)
- the response structure matches the production contract:
  - `"status": "ok"`
  - `"payload": {...}` (including `question`, `prompt`, `context`, `bugclass`, `conf`)
  - `"latency_ms"` as an integer

**Run it with:**
```
pytest tests/integration/test_rag_chat_integration.py
```
