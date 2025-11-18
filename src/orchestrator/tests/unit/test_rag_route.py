from __future__ import annotations

from api.routes import rag as rag_module
from api.schemas import RAGRequest


def test_rag_route_builds_prompt_and_calls_llm(monkeypatch):
    # Provide a fake RAG response with context
    fake_rag = {"status": "ok", "payload": {"context": "context text"}}
    monkeypatch.setattr(rag_module, "post_rag_chat", lambda req: fake_rag)

    # Fake LLM
    class DummyLLM:
        def evaluate_text(self, payload):
            return {"answer": "Here is the answer."}

    monkeypatch.setattr(rag_module, "get_llm", lambda: DummyLLM())

    req = RAGRequest(question="How to treat?", symptoms="itchy red bumps", conf=0.9, bug_class="mosquito")
    out = rag_module.orchestrator_rag(req)
    assert out["context"] == "context text"
    assert "answer" in out["llm"]
