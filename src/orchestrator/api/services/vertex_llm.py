from __future__ import annotations
from typing import Any, Dict

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

    def evaluate_text(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt = payload["prompt"]

        resp = self.client.models.generate_content(
            model=settings.VERTEX_MODEL_NAME,
            contents=prompt,
            config={"temperature": 0.2},
        )

        text = resp.text or ""

        # Always return a simple, predictable structure
        return {"answer": text}


vertex_llm_singleton: VertexLLM | None = None


def get_llm() -> VertexLLM:
    global vertex_llm_singleton
    if vertex_llm_singleton is None:
        vertex_llm_singleton = VertexLLM()
    return vertex_llm_singleton
