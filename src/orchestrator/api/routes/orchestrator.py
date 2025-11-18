from __future__ import annotations
from typing import Any, Dict

from fastapi import APIRouter

from api.services.clients import (
    post_input_eval_image,
    post_input_eval_text,
    ServiceError,
)
from api.schemas import (
    OrchestratorEvaluateRequest,
    OrchestratorEvaluateResponse,
)

router = APIRouter(prefix="/v1/orchestrator", tags=["orchestrator"])


@router.post("/evaluate", response_model=OrchestratorEvaluateResponse)
def orchestrate_evaluate(req: OrchestratorEvaluateRequest) -> Dict[str, Any]:
    """Orchestrate a frontend evaluation request.

    Expected body keys:
      - image_gcs_uri: optional
      - user_text: optional
      - overwrite_validation: bool (if true, run vl model even if eval failed)
    """
    image_uri = req.image_gcs_uri
    user_text = req.user_text
    # overwrite = bool(req.overwrite_validation)
    # TEMPORARY OVERRIDE — ALWAYS BYPASS VALIDATION
    overwrite = True
    results: Dict[str, Any] = {"image": None, "text": None}

    # Call image evaluator if present
    try:
        if image_uri:
            results["image"] = post_input_eval_image({"image_gcs_uri": image_uri})
    except ServiceError as e:
        # treat as a failed validation
        results["image"] = {"error": str(e)}

    # Call text evaluator if present
    try:
        if user_text is not None:
            txt_payload = {
                "user_text": user_text,
                "first_call": req.first_call,
                "history": req.history,
                "return_combined_text": req.return_combined_text,
                "debug": req.debug,
            }
            results["text"] = post_input_eval_text(txt_payload)
    except ServiceError as e:
        results["text"] = {"error": str(e)}

    # Decide whether to return a request for new input
    image_fail = False
    text_fail = False
    image_msg = None
    text_msg = None

    if results.get("image"):
        img = results["image"]
        if isinstance(img, dict) and (img.get("error") or img.get("usable") is False):
            image_fail = True
            image_msg = img.get("improve_message") or img.get("error")

    if results.get("text"):
        tx = results["text"]
        if isinstance(tx, dict) and (tx.get("error") or tx.get("complete") is False):
            text_fail = True
            text_msg = tx.get("improve_message") or tx.get("error")

    if (image_fail or text_fail) and not overwrite:
        # Ask frontend for better input
        detail = {
            "ok": False,
            "image_issue": image_msg,
            "text_issue": text_msg,
            "results": results,
        }
        return detail

    # If overwriting or all good, call VL model if available
    # try:
    #     vl_req = VLPredictRequest(text_raw=user_text, image_gcs=image_uri)
    #     vl_resp = post_vl_model(vl_req)
    # except ServiceError:
    #     raise HTTPException(status_code=502, detail="vl-model unavailable")

    # pred = vl_resp.prediction
    # conf = vl_resp.confidence

    # temporarily skip VLM call while the model service is offline
    pred = "mosquito"
    conf = 0.85
    print("⚠️  VLM model skipped — returning stub prediction.")

    message = f"According to our AI engine, you have been bitten by {pred} with {conf} confidence. "

    return {"ok": True, "prediction": pred, "confidence": conf, "message": message, "results": results}
