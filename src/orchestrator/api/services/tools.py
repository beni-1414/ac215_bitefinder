from typing import Optional

from api.schemas import RAGRequest
from api.services.clients import post_rag_chat, ServiceError
###
from google.genai import types
###
def get_rag_answer(
    question: str,
    symptoms: Optional[str] = None,
    bug_class: Optional[str] = None,
    conf: Optional[float] = None,
) -> str:
    """
    Tool callable by the ADK agent.
    Fetches prompt/context from the RAG service and returns a text block the model can use.
    """
    try:
        rag_resp = post_rag_chat(
            RAGRequest(
                question=question,
                symptoms=symptoms,
                bug_class=bug_class,
                conf=conf,
            )
        )
    except ServiceError as e:
        return f"[RAG unavailable: {e}]"
    except Exception as e:  # network resolution errors, etc.
        return f"[RAG call failed: {e}]"

    if rag_resp.status != "ok":
        return "[RAG returned non-ok status]"

    payload = rag_resp.payload
    context = payload.context or ""
    prompt = payload.prompt or ""

    print("RAG CALLED!!")

    # Provide both context and suggested prompt so the model can ground its answer.
    return "RAG_TOOL_OUTPUT:\n" f"Context:\n{context}\n\n" f"Prompt:\n{prompt}\n" "Use this information to answer."


###
chat_tool_decl = types.FunctionDeclaration(
    name="chat",
    description="Retrieve RAG prompt/context using bug_class + symptoms + question.",
    parameters={
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": (
                    "Optional user question. If omitted or empty, "
                    "the system will use a default 'general information about the particular bug bite' question."
                )},
            "symptoms": {"type": "string", "description": "User symptoms/notes (optional)."},
            "bug_class": {"type": "string", "description": "Detected bug class"},
            "conf": {"type": "number", "description": "Vision confidence (0-1)"},
        },
        "required": [],
        #"required": ["question"],
    },
)

bite_tools = types.Tool(function_declarations=[chat_tool_decl])

###
###
'''
# define function declarations (formerly used in gemini tools)
get_book_by_author_func = types.FunctionDeclaration(
    name="get_book_by_author",
    description="Retrieve database text chunks filtered by bug name.",
    parameters={
        "type": "object",
        "properties": {
            "bug": {
                "type": "string",
                "description": "The bug name.",
                "enum": [
                    "ant",
                    "bed bug",
                    "chigger",
                    "flea",
                    "mosquito",
                    "spider",
                    "tick",
                ],
            },
            "search_content": {
                "type": "string",
                "description": "Search text used to retrieve relevant chunks from the vector database",
            },
        },
        "required": ["bug", "search_content"],
    },
)


def get_book_by_author(bug, search_content, collection, embed_func):
    """query the vector db for bug + search_content"""
    query_embedding = embed_func(search_content)
    results = collection.query(query_embeddings=[query_embedding], n_results=10, where={"bug": bug})
    return "\n".join(results["documents"][0])


get_book_by_search_content_func = types.FunctionDeclaration(
    name="get_book_by_search_content",
    description="Retrieve database text chunks filtered only by search content.",
    parameters={
        "type": "object",
        "properties": {
            "search_content": {
                "type": "string",
                "description": "Text to use for semantic search in the vector database.",
            },
        },
        "required": ["search_content"],
    },
)


def get_book_by_search_content(search_content, collection, embed_func):
    """query the vector db for search_content only"""
    query_embedding = embed_func(search_content)
    results = collection.query(query_embeddings=[query_embedding], n_results=10)
    return "\n".join(results["documents"][0])


# define the available function toolset
cheese_expert_tool = types.Tool(
    function_declarations=[
        get_book_by_author_func,
        get_book_by_search_content_func,
    ]
)


def execute_function_calls(function_calls, collection, embed_func):
    """execute LLM-suggested function calls against your local tools"""
    parts = []

    for fc in function_calls:
        print("Function:", fc.name)

        if fc.name == "get_book_by_author":
            bug = fc.args.get("bug")
            search = fc.args.get("search_content")
            print("Calling get_book_by_author:", bug, search)

            response = get_book_by_author(bug, search, collection, embed_func)
            print("Response:", response)

            parts.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response={"content": response},
                )
            )

        elif fc.name == "get_book_by_search_content":
            search = fc.args.get("search_content")
            print("Calling get_book_by_search_content:", search)

            response = get_book_by_search_content(search, collection, embed_func)
            print("Response:", response)

            parts.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response={"content": response},
                )
            )

    return parts'''
###