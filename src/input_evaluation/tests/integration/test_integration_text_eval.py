from __future__ import annotations

from fastapi.testclient import TestClient

import api.routes.text_eval as text_route
from api.main import api


class DummyLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def evaluate_text(self, payload):
        self.calls.append(payload)
        return self.response


def test_integration_text_first_call_uses_template_and_llm(monkeypatch):
    # Use a dummy LLM to avoid calling external Vertex services
    dummy = DummyLLM(
        {
            "complete": False,
            "improve_message": "Please provide symptom details and location",
            "combined_text": None,
            "high_danger": False,
            "question_relevant": True,
            "courtesy": False,
        }
    )
    monkeypatch.setattr(text_route, "get_llm", lambda: dummy)

    client = TestClient(api)
    payload = {"user_text": "I have itchy, red bumps on my arm", "first_call": True}
    resp = client.post("/v1/evaluate/text", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["complete"] is False
    assert "improve_message" in data
    # Verify the LLM received the prompt (the DummyLLM recorded the call)
    assert dummy.calls
    prompt = dummy.calls[-1]["prompt"]
    assert "itchy" in prompt or "itch" in prompt
