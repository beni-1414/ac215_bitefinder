from __future__ import annotations

from fastapi.testclient import TestClient
from pathlib import Path
import api.routes.text_eval as text_route
from api.main import api


class DummyLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def evaluate_text(self, payload):
        self.calls.append(payload)
        return self.response


def _patch_llm(monkeypatch, response):
    dummy = DummyLLM(response)
    monkeypatch.setattr(text_route, "get_llm", lambda: dummy)
    return dummy


def _patch_template(monkeypatch):
    # Your real code calls Path(path).read_text(), so patch THAT.
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: "CONTENT:\n{content}\nEND")


def test_text_eval_first_call_suppresses_combined_text(monkeypatch):
    _patch_template(monkeypatch)

    _patch_llm(
        monkeypatch,
        {
            "complete": False,
            "improve_message": "need more info",
            "combined_text": "should not leak",
            "high_danger": False,
        },
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
    assert data["combined_text"] is None


def test_followup_defaults_missing_fields(monkeypatch):
    """If the LLM response omits 'question_relevant' or 'courtesy',
    the route should default them to False and return complete=True."""

    # Patch LLM to return a minimal response missing the keys
    dummy = DummyLLM({"improve_message": "short reply"})
    monkeypatch.setattr(text_route, "get_llm", lambda: dummy)

    # Patch template read to a stable string containing the placeholder
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: "FOLLOWUP: {user_message}")

    client = TestClient(api)
    payload = {"user_text": "is this a bite?", "first_call": False}
    resp = client.post("/v1/evaluate/text", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    # Route sets complete=True for follow-ups
    assert data["complete"] is True
    # Missing fields should default to False
    assert data["question_relevant"] is False
    assert data["courtesy"] is False
    # improve_message should come from LLM response
    assert data["improve_message"] == "short reply"
    # latency present
    assert isinstance(data["latency_ms"], int)

    # Ensure the prompt sent to LLM contains the user text
    assert dummy.calls
    assert "is this a bite?" in dummy.calls[-1]["prompt"]
