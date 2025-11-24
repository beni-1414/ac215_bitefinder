from __future__ import annotations
from typing import Any, Dict
from google.cloud import aiplatform
from google import genai
from api.config import settings
import re


class VertexLLM:
    """
    Lightweight wrapper used by the orchestrator for RAG.
    Always returns {"answer": "..."}: a plain string containing all usable info.
    """

    def __init__(self):
        aiplatform.init(project=settings.GOOGLE_CLOUD_PROJECT, location=settings.VERTEX_REGION)
        self.client = genai.Client(
            vertexai=True,
            project=settings.GOOGLE_CLOUD_PROJECT,
            location=settings.VERTEX_REGION,
        )

    def evaluate_text(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = payload["prompt"]

        resp = self.client.models.generate_content(
            model=settings.VERTEX_MODEL_NAME,
            contents=prompt,
            config={"temperature": 0.2, "response_mime_type": "text/plain"},
        )
        text = resp.text or ""

        print("===== RAW LLM RESPONSE =====")
        print(f"Response text: {text}")
        print("===========================")

        # Remove markdown code blocks if present
        if text.startswith("```"):
            # Split by code fences and take the middle part
            parts = text.split("```")
            if len(parts) >= 3:
                # Get the content between the first and last ```
                text = parts[1]
                # Remove language identifier if present (e.g., "json\n")
                if '\n' in text:
                    text = text.split('\n', 1)[1]

        cleaned_answer = re.sub(r'\*+', '', text.strip())

        if not cleaned_answer:
            cleaned_answer = "Sorry, no advice was returned."
        print(f"Final answer: {cleaned_answer}")
        return {"answer": cleaned_answer}


vertex_llm_singleton: VertexLLM | None = None


def get_llm() -> VertexLLM:
    global vertex_llm_singleton
    if vertex_llm_singleton is None:
        vertex_llm_singleton = VertexLLM()
    return vertex_llm_singleton
