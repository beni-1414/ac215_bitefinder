from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import api
import api.routes.text_eval as text_route


class DummyLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def evaluate_text(self, payload):
        self.calls.append(payload)
        return {"text": self.response}  # WRAP response under "text"


def _install_dummy_llm(monkeypatch, response):
    dummy = DummyLLM(response)
    monkeypatch.setattr(text_route, "get_llm", lambda: dummy)
    return dummy


def test_text_eval_combines_history(monkeypatch):
    dummy = _install_dummy_llm(
        monkeypatch,
        """
        {
            "complete": true,
            "improve_message": null,
            "combined_text": "merged text"
        }
        """,
    )

    client = TestClient(api)
    payload = {
        "user_text": "latest description",
        "history": ["first chunk", "second chunk"],
        "first_call": False,
        "return_combined_text": True,
    }

    resp = client.post("/v1/evaluate/text", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert data["complete"] is True
    assert data["combined_text"] == "merged text"

    # Ensure prompt contains history + current text
    prompt = dummy.calls[-1]["prompt"]
    assert "first chunk" in prompt
    assert "second chunk" in prompt
    assert "latest description" in prompt


def test_text_eval_first_call_suppresses_combined_text(monkeypatch):
    _install_dummy_llm(
        monkeypatch,
        """
        {
            "complete": false,
            "improve_message": "need more info",
            "combined_text": "should not leak"
        }
        """,
    )

    client = TestClient(api)
    payload = {
        "user_text": "single description",
        "history": [],
        "first_call": True,
        "return_combined_text": True,
    }

    resp = client.post("/v1/evaluate/text", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert data["complete"] is False
    assert data["improve_message"] == "need more info"
    assert data["combined_text"] is None  # suppressed on first call
