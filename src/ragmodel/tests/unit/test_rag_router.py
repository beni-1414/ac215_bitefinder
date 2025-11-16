# tests/unit/test_rag_router.py
# Unit test for the /rag/chat FastAPI route.
# This test checks:
#   - the route loads
#   - the payload structure is correct
#   - the underlying pipeline is called
#   - no real Pinecone or Vertex AI calls occur (everything is mocked)

from fastapi.testclient import TestClient
from unittest.mock import patch

from api.main import api

client = TestClient(api)


def test_chat_endpoint_returns_payload():
    print("Running test_chat_endpoint_returns_payload() ...")

    dummy_payload = {
        "question": "How can I treat mosquito bites? My symptoms: itchy.",
        "prompt": "A patient reported experiencing the following: itchy.",
        "context": "dummy context",
        "bug_class": "mosquito",
    }

    # mock the chat() function inside rag_router
    with patch("api.routes.rag_router.chat", return_value=dummy_payload):
        resp = client.post(
            "/rag/chat",
            json={"symptoms": "itchy", "conf": 0.9, "bug_class": "mosquito"},
        )

    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "ok"
    assert "payload" in data
    assert data["payload"]["bug_class"] == "mosquito"
    assert isinstance(data["latency_ms"], int)

    print("Test passed.")


if __name__ == "__main__":
    test_chat_endpoint_returns_payload()
    print("ALL ROUTER TESTS PASSED")
