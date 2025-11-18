from __future__ import annotations

from types import SimpleNamespace
import json

import pytest

import api.services.clients as clients
from api.schemas import VLPredictRequest, RAGRequest


class DummyResponse:
    def __init__(self, status_code: int, json_data):
        self.status_code = status_code
        self._json = json_data
        # Provide a .text attribute like httpx.Response has
        try:
            self.text = json.dumps(json_data)
        except Exception:
            self.text = str(json_data)

    def json(self):
        return self._json


class DummyClient:
    def __init__(self, resp: DummyResponse):
        self.resp = resp

    def post(self, url, json=None):
        return self.resp

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_post_vl_model_success(monkeypatch):
    resp = DummyResponse(200, {"prediction": "mosquito", "confidence": 0.87})
    monkeypatch.setattr(clients, "httpx", SimpleNamespace(Client=lambda timeout=None: DummyClient(resp)))

    req = VLPredictRequest(text_raw="hi")
    out = clients.post_vl_model(req)
    assert out.prediction == "mosquito"
    assert out.confidence == 0.87


def test_post_vl_model_failure(monkeypatch):
    resp = DummyResponse(500, {"error": "fail"})
    monkeypatch.setattr(clients, "httpx", SimpleNamespace(Client=lambda timeout=None: DummyClient(resp)))
    req = VLPredictRequest()
    with pytest.raises(clients.ServiceError):
        clients.post_vl_model(req)


def test_post_rag_chat_success(monkeypatch):
    rag_json = {
        "status": "ok",
        "payload": {"context": "useful info", "prompt": "p", "question": "Q", "bug_class": None, "conf": 0.0},
        "latency_ms": 10,
    }
    resp = DummyResponse(200, rag_json)
    monkeypatch.setattr(clients, "httpx", SimpleNamespace(Client=lambda timeout=None: DummyClient(resp)))

    req = RAGRequest(question="Q")
    out = clients.post_rag_chat(req)
    assert out.context == "useful info"
