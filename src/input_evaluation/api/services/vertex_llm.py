from __future__ import annotations
import json
from typing import Any, Dict

from google.cloud import aiplatform
from google import genai
from api.config import settings


class VertexLLM:
    def __init__(self):
        aiplatform.init(project=settings.GOOGLE_CLOUD_PROJECT, location=settings.VERTEX_REGION)
        self.client = genai.Client(
            vertexai=True,
            project=settings.GOOGLE_CLOUD_PROJECT,
            location=settings.VERTEX_REGION,
        )

    def evaluate_text(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Payload contains fields you choose in your prompt template.
        Returns whatever JSON the model outputs.
        """
        prompt = payload["prompt"]
        resp = self.client.models.generate_content(
            model=settings.VERTEX_MODEL_NAME,
            contents=prompt,
            config={
                "temperature": 0.2,
                "response_mime_type": "application/json",
            },
        )
        text = resp.text or "{}"
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Fallback minimal structure
            data = {
                "complete": False,
                "improve_message": "Error processing request.",
                "combined_text": None,
            }
        return data  # <-- This closes the function


vertex_llm_singleton: VertexLLM | None = None


def get_llm() -> VertexLLM:
    global vertex_llm_singleton
    if vertex_llm_singleton is None:
        vertex_llm_singleton = VertexLLM()
    return vertex_llm_singleton
