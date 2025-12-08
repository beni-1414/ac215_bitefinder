from __future__ import annotations
from typing import Any, Dict

import httpx

from api.config import settings
from api.schemas import VLPredictRequestGCS, VLPredictRequestBase64, VLPredictResponse, RAGRequest, RAGModelResponse


class ServiceError(RuntimeError):
    pass

#15.0 --> 60.0
def post_input_eval_text(payload: Dict[str, Any], timeout: float = 60.0) -> Dict[str, Any]:
    url = f"{settings.INPUT_EVAL_URL}/v1/evaluate/text"
    with httpx.Client(timeout=timeout) as c:
        r = c.post(url, json=payload)
    if r.status_code >= 400:
        raise ServiceError(f"input-eval text error {r.status_code}: {r.text}")
    return r.json()

#20.0 --> 120.0
def post_input_eval_image(payload: Dict[str, Any], timeout: float = 120.0) -> Dict[str, Any]:
    url = f"{settings.INPUT_EVAL_URL}/v1/evaluate/image"
    with httpx.Client(timeout=timeout) as c:
        r = c.post(url, json=payload)
    if r.status_code >= 400:
        raise ServiceError(f"input-eval image error {r.status_code}: {r.text}")
    return r.json()

#10.0 --> 100.0
def post_vl_model(req: VLPredictRequestBase64 | VLPredictRequestGCS, timeout: float = 100.0) -> VLPredictResponse:
    """Call the VL model endpoint. Accepts a VLPredictRequestGCS and returns a VLPredictResponse."""
    url = f"{settings.VL_MODEL_URL}/vlmodel/predict"
    with httpx.Client(timeout=timeout) as c:
        r = c.post(url, json=req.model_dump())
    if r.status_code >= 400:
        raise ServiceError(f"vl-model error {r.status_code}: {r.text}")
    return VLPredictResponse.model_validate(r.json())


def post_rag_chat(req: RAGRequest, timeout: float = 100.0) -> RAGModelResponse:
    url = f"{settings.RAG_MODEL_URL}/rag/chat"
    with httpx.Client(timeout=timeout) as c:
        #payload = req.model_dump()
        payload = req.model_dump(exclude_none=True)  # drop Nones


        # normalize confidence
        if payload.get("conf") is None:
            payload["conf"] = 0.0

        try:
            payload["conf"] = float(payload["conf"])
        except Exception:
            payload["conf"] = 0.0

        r = c.post(url, json=payload)

    if r.status_code >= 400:
        raise ServiceError(f"rag-model error {r.status_code}: {r.text}")

    return RAGModelResponse.model_validate(r.json())

###
def post_rag_search_by_bug(payload: dict, timeout: float = 10.0) -> RAGModelWrapper:
    url = f"{settings.RAG_MODEL_URL}/rag/search_by_bug"
    with httpx.Client(timeout=timeout) as c:
        r = c.post(url, json=payload)
    if r.status_code >= 400:
        raise ServiceError(f"rag-model search_by_bug error {r.status_code}: {r.text}")
    return RAGModelWrapper.model_validate(r.json())

def post_rag_search_by_symptom(payload: dict, timeout: float = 10.0) -> RAGModelWrapper:
    url = f"{settings.RAG_MODEL_URL}/rag/search_by_symptom"
    with httpx.Client(timeout=timeout) as c:
        r = c.post(url, json=payload)
    if r.status_code >= 400:
        raise ServiceError(f"rag-model search_by_symptom error {r.status_code}: {r.text}")
    return RAGModelWrapper.model_validate(r.json())
###