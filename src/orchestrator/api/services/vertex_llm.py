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
        Expect the model to return strict JSON with keys: complete, improve_message, combined_text.
        """
        prompt = payload["prompt"]
        # Enforce JSON-only output
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
            data = {"answer": "There was an error parsing the model response."}
        # Ensure keys exist
        return {"answer": data.get("answer", "No answer provided by the LLM model.")}


vertex_llm_singleton: VertexLLM | None = None


def get_llm() -> VertexLLM:
    global vertex_llm_singleton
    if vertex_llm_singleton is None:
        vertex_llm_singleton = VertexLLM()
    return vertex_llm_singleton
