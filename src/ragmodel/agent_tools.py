import json
import os
from google import genai
from google.genai import types

# initialize vertex ai genai client
GCP_PROJECT = os.environ["GCP_PROJECT"]
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")

client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)

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
                "description": "Search text used to retrieve relevant chunks from the vector database. The search term is embedded and compared against stored document embeddings using cosine similarity.",
            },
        },
        "required": ["bug", "search_content"],
    },
)


def get_book_by_author(bug, search_content, collection, embed_func):
    """query the vector db for bug + search_content"""
    query_embedding = embed_func(search_content)
    results = collection.query(
        query_embeddings=[query_embedding], n_results=10, where={"bug": bug}
    )
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

    return parts
