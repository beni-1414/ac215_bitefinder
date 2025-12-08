from __future__ import annotations
import os
from typing import List, Optional
from google import genai
from google.genai import types

#from api.services.clients import post_rag_search_by_bug, post_rag_search_by_symptom
from api.services.tools import bite_tools, get_rag_answer
from api.schemas import RAGModelWrapper

GCP_PROJECT = os.environ["GCP_PROJECT"]
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
gen_client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
#generative_model = gen_client.models.get_model("gemini-2.0-flash-001")

MODEL_NAME = "gemini-2.0-flash-001"

def execute_function_calls(function_calls: List[types.FunctionCall]) -> List[types.Part]:
    parts = []
    for fc in function_calls:
        if fc.name != "chat":
            continue
        context = get_rag_answer(
            question=fc.args.get("question"),
            symptoms=fc.args.get("symptoms"),
            bug_class=fc.args.get("bug_class"),
            conf=fc.args.get("conf"),
        )
        parts.append(types.Part.from_function_response(name=fc.name, response={"content": context}))
    return parts

def run_agent(user_question: str, bug_class: Optional[str], conf: Optional[float]) -> str:
    hint = ""
    if conf is not None:
        hint = f"Model confidence: {conf:.2f}. Bug: {bug_class or 'unknown'}.\n"

    user_prompt = types.Content(
        role="user",
        parts=[
            types.Part.from_text(
                text=(
                    hint
                    + f"Question: {user_question}\n"
                    f"Detected bug: {bug_class or 'unknown'}\n"
                    "Call the chat tool with the question/symptoms/bug_class/conf as needed, then answer concisely."
                )
            )
        ],
    )

    step1 = gen_client.models.generate_content(
        model=MODEL_NAME,
        contents=[user_prompt],
        config=types.GenerateContentConfig(
            temperature=0,
            tools=[bite_tools],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )

    function_calls = []
    for part in step1.candidates[0].content.parts:
        fc = getattr(part, "function_call", None)
        if fc:
            function_calls.append(fc)

    function_responses = execute_function_calls(function_calls)

    final = gen_client.models.generate_content(
        model=MODEL_NAME,
        contents=[user_prompt, step1.candidates[0].content, types.Content(parts=function_responses)],
        config=types.GenerateContentConfig(
            temperature=0,
            tools=[bite_tools],
        ),
    )

    answer = ""
    for part in final.candidates[0].content.parts:
        if getattr(part, "text", None):
            answer += part.text
    return answer.strip()
'''
search_by_bug_decl = types.FunctionDeclaration(
    name="search_by_bug",
    description="Retrieve Pinecone chunks filtered by bug name.",
    parameters={
        "type": "object",
        "properties": {
            "bug": {"type": "string", "description": "Bug name"},
            "search_content": {"type": "string", "description": "User question/symptoms"},
            "question": {"type": "string", "description": "Optional user question override"},
            "conf": {"type": "number", "description": "Vision confidence"},
        },
        "required": ["bug", "search_content"],
    },
)
search_by_symptom_decl = types.FunctionDeclaration(
    name="search_by_symptom",
    description="Retrieve Pinecone chunks using only question/symptoms.",
    parameters={
        "type": "object",
        "properties": {
            "search_content": {"type": "string", "description": "User question or symptoms"},
            "question": {"type": "string", "description": "Optional user question override"},
        },
        "required": ["search_content"],
    },
)
bite_tools = types.Tool(function_declarations=[search_by_bug_decl, search_by_symptom_decl])

def _call_rag_bug(bug: str, search_content: str, question: Optional[str], conf: Optional[float]) -> str:
    # rag_router expects `bug`, not `bug_class`
    resp: RAGModelWrapper = post_rag_search_by_bug(
        {
            "bug": bug,
            "search_content": search_content,
            "question": question or search_content,
            "conf": conf,
        }
    )
    return (resp.payload.context or "") if resp.status == "ok" else ""

def _call_rag_symptom(search_content: str, question: Optional[str]) -> str:
    resp: RAGModelWrapper = post_rag_search_by_symptom(
        {
            "search_content": search_content,
            "question": question or search_content,
        }
    )
    return (resp.payload.context or "") if resp.status == "ok" else ""

def execute_function_calls(function_calls: List[types.FunctionCall]) -> List[types.Part]:
    parts = []
    for fc in function_calls:
        if fc.name == "search_by_bug":
            context = _call_rag_bug(
                bug=fc.args.get("bug"),
                search_content=fc.args.get("search_content"),
                question=fc.args.get("question"),
                conf=fc.args.get("conf"),
            )
        elif fc.name == "search_by_symptom":
            context = _call_rag_symptom(
                search_content=fc.args.get("search_content"),
                question=fc.args.get("question"),
            )
        else:
            continue
        parts.append(types.Part.from_function_response(name=fc.name, response={"content": context}))
    return parts

def run_agent(user_question: str, bug_class: Optional[str], conf: Optional[float]) -> str:
    hint = ""
    if conf is not None:
        hint = (
            f"Model confidence: {conf:.2f}. If confidence is high and bug is known ('{bug_class}'), "
            "prefer search_by_bug; otherwise prefer search_by_symptom.\n"
        )

    user_prompt = types.Content(
    role="user",
    parts=[types.Part.from_text(
        text=(
            hint +
            f"Question: {user_question}\n"
            f"Detected bug: {bug_class or 'unknown'}\n"
            "Decide which tool to call and then answer concisely."
        )
    )],
)


    step1 = gen_client.models.generate_content(
        model=MODEL_NAME,
        contents=[user_prompt],
        config=types.GenerateContentConfig(
            temperature=0,
            tools=[bite_tools],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )

    #function_calls = step1.candidates[0].function_calls
    # Pull tool calls from the parts on the first candidate
    function_calls = []
    for part in step1.candidates[0].content.parts:
        fc = getattr(part, "function_call", None)
        if fc:
            function_calls.append(fc)

    function_responses = execute_function_calls(function_calls)

    final = gen_client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            user_prompt,
            step1.candidates[0].content,
            types.Content(parts=function_responses),
        ],
        tools=[bite_tools],
    )


    answer = ""
    for part in final.candidates[0].content.parts:
        if getattr(part, "text", None):
            answer += part.text
    return answer.strip()
'''
