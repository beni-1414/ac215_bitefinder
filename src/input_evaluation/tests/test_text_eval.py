from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
import app.routes.text_eval as text_route


class DummyLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def evaluate_text(self, payload):
        self.calls.append(payload)
        return self.response


def _install_dummy_llm(monkeypatch, response):
    dummy = DummyLLM(response)
    monkeypatch.setattr(text_route, "get_llm", lambda: dummy)
    return dummy


def test_text_eval_combines_history(monkeypatch):
    dummy = _install_dummy_llm(
        monkeypatch,
        {
            "complete": True,
            "improve_message": None,
            "combined_text": "merged text",
        },
    )

    client = TestClient(app)
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
    prompt = dummy.calls[-1]["prompt"]
    assert "first chunk" in prompt and "second chunk" in prompt
    assert "latest description" in prompt


def test_text_eval_first_call_suppresses_combined_text(monkeypatch):
    _install_dummy_llm(
        monkeypatch,
        {
            "complete": False,
            "improve_message": "need more info",
            "combined_text": "should not leak",
        },
    )

    client = TestClient(app)
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
    assert data["combined_text"] is None
