import json
import google.generativeai as genai
from google.generativeai import types
import os
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# import vertexai
# from vertexai.generative_models import FunctionDeclaration, Tool, Part

# Specify a function declaration and parameters for an API request
get_book_by_author_func = types.FunctionDeclaration(
    name="get_book_by_author",
    description="Get the book chunks filtered by bug name",
    # Function parameters are specified in OpenAPI JSON schema format
    parameters={
        "type": "object",
        "properties": {
            "bug": {
                "type": "string",
                "description": "The bug name",
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
                "description": "The search text to filter content from data bases. The search term is compared against the data base text based on cosine similarity. Expand the search term to a a sentence or two to get better matches",
            },
        },
        "required": ["bug", "search_content"],
    },
)

# book --> 'type' of bite, author --> bug
def get_book_by_author(bug, search_content, collection, embed_func):

    query_embedding = embed_func(search_content)

    # Query based on embedding value
    results = collection.query(
        query_embeddings=[query_embedding], n_results=10, where={"bug": bug}
    )
    return "\n".join(results["documents"][0])


get_book_by_search_content_func = types.FunctionDeclaration(
    name="get_book_by_search_content",
    description="Get the book chunks filtered by search terms",
    # Function parameters are specified in OpenAPI JSON schema format
    parameters={
        "type": "object",
        "properties": {
            "search_content": {
                "type": "string",
                "description": "The search text to filter content from the data base. The search term is compared against the data base text based on cosine similarity. Expand the search term to a a sentence or two to get better matches",
            },
        },
        "required": ["search_content"],
    },
)


def get_book_by_search_content(search_content, collection, embed_func):

    query_embedding = embed_func(search_content)

    # Query based on embedding value
    results = collection.query(query_embeddings=[query_embedding], n_results=10)
    return "\n".join(results["documents"][0])


# Define all functions available to the cheese expert
cheese_expert_tool = types.Tool(
    function_declarations=[get_book_by_author_func, get_book_by_search_content_func]
)


def execute_function_calls(function_calls, collection, embed_func):
    parts = []
    for function_call in function_calls:
        print("Function:", function_call.name)
        if function_call.name == "get_book_by_author":
            print(
                "Calling function with args:",
                function_call.args["bug"],
                function_call.args["search_content"],
            )
            response = get_book_by_author(
                function_call.args["bug"],
                function_call.args["search_content"],
                collection,
                embed_func,
            )
            print("Response:", response)
            # function_responses.append({"function_name":function_call.name, "response": response})
            parts.append(
                types.Part.from_function_response(
                    name=function_call.name,
                    response={
                        "content": response,
                    },
                ),
            )
        if function_call.name == "get_book_by_search_content":
            print("Calling function with args:", function_call.args["search_content"])
            response = get_book_by_search_content(
                function_call.args["search_content"], collection, embed_func
            )
            print("Response:", response)
            # function_responses.append({"function_name":function_call.name, "response": response})
            parts.append(
                types.Part.from_function_response(
                    name=function_call.name,
                    response={
                        "content": response,
                    },
                ),
            )

    return parts
