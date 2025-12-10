# Agent API

**NOT USED IN PRODUCTION, AGENT IMPLEMENTATION IS DIRECTLY INTEGRATED WITH ORCHESTRATOR**. This is a development container based off the APCOMP 215 tutorial.

## llm_rag_chat
This provides a Retrieval-Augmented Generation (RAG) pipeline used to answer bug bite–related questions. It manages user chat sessions, embeds queries, retrieves context from Pinecone, and generates responses using Gemini. The system exposes REST API endpoints for creating chats, continuing conversations, and retrieving past sessions.


| File                                | Description                                                                                                                                                                    |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `api/routers/llm_rag_chat.py`       | Defines the FastAPI routes for the LLM chat system (`/chats`, `/chats/{chat_id}`). Manages session IDs, initializes or resumes chats, and forwards messages to the RAG engine. |
| `api/utils/llm_rag_utils.py`        | Implements the full RAG workflow: building prompts, retrieving Pinecone context, handling images, generating responses with Gemini, and maintaining conversation history.      |
| `api/services/pinecone_adapter.py`  | Handles low-level Pinecone operations: upserting embeddings, querying by vector, and retrieving context chunks for RAG.                                                        |

## How it works

### Run the ragmodel db Container first

### Build & Run Container
```
sh docker-shell.sh
```

### Inside container
Run the following command within the docker shell:
```
uvicorn_server
```
### User Input
The frontend sends:
```
POST /chats
Header: X-Session-ID: <some_user_id>
Body:
{
  "content": "My arm has a red itchy bump...",
  "image": null
}
```

### llm_rag_chat.py – API Layer & Session Manager
This module:
- Creates a new chat session (or loads an existing one)
- Adds the user message to the conversation
- Calls generate_chat_response() from llm_rag_utils.py
- Saves the updated conversation
- Returns the full assistant response + chat metadata


### Test with curl
```
curl -X 'POST' \
  'http://localhost:9000/llm-rag/chats' \
  -H 'accept: application/json' \
  -H 'X-Session-ID: test123' \
  -H 'Content-Type: application/json' \
  -d '{
  "content": "What kind of bug bite leaves a small red dot and itches a lot?",
  "image": "",
  "message_id": "string",
  "role": "string"
}'
```
