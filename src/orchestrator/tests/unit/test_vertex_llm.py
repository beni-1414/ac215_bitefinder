from __future__ import annotations

import api.services.vertex_llm as vv


class DummyModels:
    def __init__(self, text):
        self._text = text

    def generate_content(self, model=None, contents=None, config=None):
        return type("R", (), {"text": self._text})()


class DummyClient:
    def __init__(self, models):
        self.models = models


def test_evaluate_text_parses_json(monkeypatch):
    monkeypatch.setattr(vv, "aiplatform", type("A", (), {"init": lambda *a, **k: None}))
    monkeypatch.setattr(
        vv,
        "genai",
        type("G", (), {"Client": lambda *a, **k: DummyClient(DummyModels('{"answer":"ok"}'))}),
    )

    inst = vv.VertexLLM()
    out = inst.evaluate_text({"prompt": "p"})

    # FIX: out is already a dict
    assert out.get("answer") == "ok"


def test_evaluate_text_handles_invalid_json(monkeypatch):
    monkeypatch.setattr(vv, "aiplatform", type("A", (), {"init": lambda *a, **k: None}))
    monkeypatch.setattr(vv, "genai", type("G", (), {"Client": lambda *a, **k: DummyClient(DummyModels("not-json"))}))

    inst = vv.VertexLLM()
    out = inst.evaluate_text({"prompt": "p"})
    assert "answer" in out
