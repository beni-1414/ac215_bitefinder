# tests/test_rag_agent.py

from fastapi.testclient import TestClient
import pytest

from api.main import api


@pytest.fixture(autouse=True)
def fake_gcp_env(monkeypatch):
    # So get_firestore_client() doesn’t blow up if it’s accidentally called
    monkeypatch.setenv("GCP_PROJECT", "test-project")


@pytest.fixture(autouse=True)
def stub_langchain_agent(monkeypatch):
    # We want to intercept get_langchain_agent() before it builds a real one
    from api.routes.rag_agent import get_langchain_agent  # noqa: F401

    class FakeAgent:
        def __init__(self):
            self.calls = []

        def query(self, input: str, config: dict):
            self.calls.append({"input": input, "config": config})
            # Simulate LangchainAgent response shape
            return {"input": input, "output": "This is a test answer about bug bites."}

    fake = FakeAgent()

    def fake_get_langchain_agent():
        return fake

    # Patch the factory function used by run_rag_agent
    monkeypatch.setattr("api.routes.rag_agent.get_langchain_agent", fake_get_langchain_agent)

    return fake


def test_run_rag_agent_happy_path(stub_langchain_agent):
    client = TestClient(api)

    payload = {
        "question": "My arm is swollen after a mosquito bite, what should I do?",
        "symptoms": "Red, itchy, slightly warm to touch",
        "bug_class": "mosquito",
        "conf": 0.9,
    }

    resp = client.post("/v1/orchestrator/rag-agent", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert data["llm"] == "This is a test answer about bug bites."
    assert "session_id" in data
    assert data["user_id"] == "anonymous"

    # Optional: inspect what we sent into the agent
    assert len(stub_langchain_agent.calls) == 1
    call = stub_langchain_agent.calls[0]
    assert "User question:" in call["input"]
    assert "mosquito" in call["input"]
    assert call["config"]["configurable"]["session_id"] == data["session_id"]


def test_run_rag_agent_missing_question():
    client = TestClient(api)

    payload = {
        "question": "",
        "symptoms": "something",
        "bug_class": "mosquito",
        "conf": 0.9,
    }

    resp = client.post("/v1/orchestrator/rag-agent", json=payload)
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Question is required."
