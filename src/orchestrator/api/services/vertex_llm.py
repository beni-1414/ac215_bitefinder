from __future__ import annotations
from typing import Any, Dict
import json
from google.cloud import aiplatform
from google import genai
from api.config import settings


class VertexLLM:
    """
    Lightweight wrapper used by the orchestrator for RAG.
    Returns plain text -> {"answer": "..."}.
    """

    def __init__(self):
        aiplatform.init(project=settings.GOOGLE_CLOUD_PROJECT, location=settings.VERTEX_REGION)
        self.client = genai.Client(
            vertexai=True,
            project=settings.GOOGLE_CLOUD_PROJECT,
            location=settings.VERTEX_REGION,
        )

    import json

    def evaluate_text(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = payload["prompt"]

        resp = self.client.models.generate_content(
            model=settings.VERTEX_MODEL_NAME,
            contents=prompt,
            config={"temperature": 0.2, "response_mime_type": "application/json"},
        )

        text = resp.text or ""

        print("===== RAW LLM RESPONSE =====")
        print(f"Response text: {text}")
        print("===========================")

        # Parse the JSON response
        try:
            # Remove markdown code blocks if present
            if text.startswith("```json"):
                text = text.split("```json")[1].split("```")[0].strip()
            elif text.startswith("```"):
                text = text.split("```")[1].split("```")[0].strip()

            result = json.loads(text)
            print(f"Parsed result: {result}")
            return result
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to parse LLM JSON response: {text}")
            print(f"Error: {e}")
            return {}


vertex_llm_singleton: VertexLLM | None = None


def get_llm() -> VertexLLM:
    global vertex_llm_singleton
    if vertex_llm_singleton is None:
        vertex_llm_singleton = VertexLLM()
    return vertex_llm_singleton
