# Orchestrator

FastAPI service that ties together BiteFinder components. All routes are mounted under `/v1/orchestrator`.

## Evaluate endpoint
- `POST /evaluate` accepts `OrchestratorEvaluateRequest` (image as `image_gcs_uri` or `image_base64`, optional `user_text`, `first_call`, `history`, etc.).
- Runs input evaluators first (`input_evaluation` service) and short-circuits with improvement hints unless the caller sets `overwrite_validation=true`.
- If inputs pass (or are overwritten), forwards the image + text to the VL model (`vlmodel` service) and returns `ok`, `prediction`, `confidence`, and the raw eval results.

## Hardcoded RAG endpoint
- `POST /rag` accepts `RAGRequest` (`question`, optional `symptoms`, `bug_class`, `conf`, `session_id`).
- Calls the downstream RAG model (`RAG_MODEL_URL` -> `/rag/chat`) to get a suggested prompt/context, then runs a Vertex model to produce the final answer.
- If the RAG service returns an empty prompt, it builds a fallback prompt from the request so the LLM can still reply.
- Response includes the retrieved `context` and the Vertex model output under `llm`.

## RAG agent endpoint
- `POST /rag-agent` also takes `RAGRequest` so clients can switch between `/rag` and `/rag-agent` without changing payloads.
- Wraps a Google ADK agent (configured with a BiteFinder system prompt) that can decide when to call the same RAG tool (`get_rag_answer`), which in turn hits the RAG model service.
- Streams internally via ADK and returns a single concise `answer` plus `session_id`/`user_id` for follow-up turns.

## Interchangeability
Both `/rag` and `/rag-agent` consume the exact same schema (`RAGRequest`), so the frontend can toggle between a direct RAG call and the agent-powered flow without changing request construction. The agent’s tool calls the same RAG model as the hardcoded endpoint to keep grounding consistent. (Ignore the `/rag/agent` route in `rag.py`; it is a work in progress.)
