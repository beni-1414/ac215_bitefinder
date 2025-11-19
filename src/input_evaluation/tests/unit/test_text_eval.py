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
        return self.response


def install_dummy_llm(monkeypatch, response):
    dummy = DummyLLM(response)
    monkeypatch.setattr(text_route, "get_llm", lambda: dummy)
    return dummy


# ============================================================================
# FIXED TEST 1: test_text_eval_combines_history
# ============================================================================
def test_text_eval_combines_history(monkeypatch):
    """
    IMPORTANT CHANGE:
    The dummy LLM returns a *simple safe dict* so that template.format()
    never interprets `"complete"` as a template key.
    """
    dummy = install_dummy_llm(
        monkeypatch,
        {
            "complete": True,
            "improve_message": None,
            "combined_text": "merged text",
        },
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

    # Expected downstream output
    assert data["complete"] is True
    assert data["combined_text"] == "merged text"

    # Check that our prompt construction included history + latest text
    sent_prompt = dummy.calls[-1]["prompt"]
    assert "first chunk" in sent_prompt
    assert "second chunk" in sent_prompt
    assert "latest description" in sent_prompt


# ============================================================================
# FIXED TEST 2: test_text_eval_first_call_suppresses_combined_text
# ============================================================================
def test_text_eval_first_call_suppresses_combined_text(monkeypatch):
    install_dummy_llm(
        monkeypatch,
        {
            "complete": False,
            "improve_message": "need more info",
            "combined_text": "ignore this",
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

    # Because first_call=True, combined_text MUST be suppressed
    assert data["combined_text"] is None
