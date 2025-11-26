from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import api
from api.routes import orchestrator as orchestrator_module
from api.schemas import VLPredictResponse


client = TestClient(api)


def test_evaluate_success_prediction(monkeypatch):
    # Image and text evaluators return OK; VL model returns prediction
    monkeypatch.setattr(
        orchestrator_module,
        "post_input_eval_image",
        lambda payload: {"usable": True},
    )

    monkeypatch.setattr(
        orchestrator_module,
        "post_input_eval_text",
        lambda payload: {"complete": True},
    )

    # Return a VLPredictResponse-like object
    monkeypatch.setattr(
        orchestrator_module,
        "post_vl_model",
        lambda req: VLPredictResponse(prediction="tick", confidence=0.83),
    )

    payload = {
        "image_gcs_uri": "gs://bucket/image.jpg",
        "first_call": True,
    }

    r = client.post("/v1/orchestrator/evaluate", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("ok") is True
    assert data.get("prediction") == "tick"
    assert "confidence" in data


def test_evaluate_followup_returns_eval_only(monkeypatch):
    # For follow-up (first_call=False) orchestrator should return evaluation-only
    monkeypatch.setattr(
        orchestrator_module,
        "post_input_eval_text",
        lambda payload: {"eval": {"question_relevant": True, "improve_message": None, "courtesy": False, "complete": True}},
    )

    payload = {"user_text": "Followup?", "first_call": False}
    r = client.post("/v1/orchestrator/evaluate", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("ok") is True
    assert isinstance(data.get("eval"), dict)
    assert data["eval"]["question_relevant"] is True


def test_evaluate_requests_better_input_when_image_unusable(monkeypatch):
    # Image evaluator reports unusable; orchestrator should ask for better input
    monkeypatch.setattr(
        orchestrator_module,
        "post_input_eval_image",
        lambda payload: {"usable": False, "improve_message": "blurry image"},
    )

    monkeypatch.setattr(
        orchestrator_module,
        "post_input_eval_text",
        lambda payload: {"complete": True},
    )

    payload = {"image_gcs_uri": "gs://bucket/bad.jpg", "user_text": "hi", "first_call": True}
    r = client.post("/v1/orchestrator/evaluate", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("ok") is False
    assert "image_issue" in data or (data.get("results") and data["results"].get("image"))
