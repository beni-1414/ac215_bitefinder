from __future__ import annotations
from typing import Any, Dict

from fastapi import APIRouter

from api.services.clients import (
    post_input_eval_image,
    post_input_eval_text,
    post_vl_model,
    ServiceError,
)
from api.schemas import (
    OrchestratorEvaluateRequest,
    OrchestratorEvaluateResponse,
    VLPredictRequest,
)

router = APIRouter(prefix="/v1/orchestrator", tags=["orchestrator"])


@router.post("/evaluate", response_model=OrchestratorEvaluateResponse)
def orchestrate_evaluate(req: OrchestratorEvaluateRequest) -> Dict[str, Any]:
    """Orchestrate a frontend evaluation request."""
    image_uri = req.image_gcs_uri
    user_text = req.user_text
    print("TESTING")
    print(image_uri)
    print(user_text)
    overwrite = bool(req.overwrite_validation)
    results: Dict[str, Any] = {"image": None, "text": None}

    # Call image evaluator if present
    try:
        if image_uri:
            results["image"] = post_input_eval_image({"image_gcs_uri": image_uri})
    except ServiceError as e:
        results["image"] = {"error": str(e)}
    print("EVAL RESULTS:", results)

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
    print("EVAL RESULTS:", results)

    # =========================================
    # FOR FOLLOWUP QUESTIONS: RETURN EVAL ONLY
    # =========================================
    if not req.first_call:
        textblock = results.get("text")
        print("FOLLOWUP RAW textblock:", textblock)

        # Flatten deeply
        while isinstance(textblock, dict) and "eval" in textblock and isinstance(textblock["eval"], dict):
            textblock = textblock["eval"]

        response = {
            "ok": True,
            "eval": {
                "question_relevant": textblock.get("question_relevant", False),
                "improve_message": textblock.get("improve_message"),
                "courtesy": textblock.get("courtesy", False),
                "complete": textblock.get("complete", True),
            },
        }
        print("FOLLOWUP RESPONSE TO FRONTEND:", response)
        return response

    # =========================================
    # FOR INITIAL MESSAGES: CONTINUE WITH PREDICTION
    # =========================================

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

    # Call VL model and return prediction
    vl_model_req = VLPredictRequest(
        image_gcs='user-input/image.jpg',  # TODO: THIS SHOULD NOT BE HARDCODED (IMAGE GCS URL IS CURRENTLY NONE)
        text_raw=user_text,
    )
    try:
        vl_model_res = post_vl_model(vl_model_req)
    except ServiceError as e:
        return {
            "ok": False,
            "prediction": None,
            "confidence": None,
            "message": f"VL model prediction service error: {str(e)}",
            "results": results,
        }

    pred = vl_model_res.prediction
    conf = vl_model_res.confidence

    message = f"According to our AI engine, you have been bitten by {pred} with {conf} confidence. "

    return {"ok": True, "prediction": pred, "confidence": conf, "message": message, "results": results}
