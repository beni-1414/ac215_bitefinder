from __future__ import annotations

from api import schemas


def test_orchestrator_evaluate_request_defaults():
    req = schemas.OrchestratorEvaluateRequest()
    assert req.image_gcs_uri is None
    assert req.user_text is None
    assert req.overwrite_validation is False
    assert req.first_call is True


def test_vl_and_rag_schemas_validation():
    vl = schemas.VLPredictRequestGCS(text_raw="hi", image_gcs="gs://b/x.jpg")
    assert vl.text_raw == "hi"
    rag = schemas.RAGRequest(question="Q", symptoms="itchy", conf=0.8, bug_class="mosquito")
    assert rag.question == "Q"
    assert rag.conf == 0.8
