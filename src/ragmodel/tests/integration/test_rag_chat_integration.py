from fastapi.testclient import TestClient
from unittest.mock import patch

from api.main import api

client = TestClient(api)


def test_rag_chat_integration_builds_payload_and_returns_ok():
    dummy_payload = {
        "question": "How can I treat mosquito bites? My symptoms: itchy red bumps.",
        "prompt": "User question about mosquito treatment and symptoms.",
        "context": "chunk1 text\nchunk2...",
        "bugclass": "mosquito",
        "conf": 0.9,
    }

    # Patch the chat function where the router imports it
    with patch("api.routes.rag_router.chat", return_value=dummy_payload):
        body = {
            "symptoms": "itchy red bumps",
            "conf": 0.9,
            "bug_class": "mosquito",  # <-- must be bug_class
            "question": "How can I treat mosquito bites?",
        }

        resp = client.post("/rag/chat", json=body)
        assert resp.status_code == 200, resp.text

        data = resp.json()
        assert data["status"] == "ok"
        assert "payload" in data
        assert isinstance(data["latency_ms"], int)

        payload = data["payload"]
        # payload is whatever chat returns; you can assert on its shape/content
        assert payload["bugclass"] == "mosquito"
        assert payload["conf"] == 0.9
        assert "itchy" in payload["question"]
        assert "mosquito" in payload["prompt"]
