from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional


# Text
class TextEvalRequest(BaseModel):
    user_text: str
    first_call: bool = True
    history: List[str] = Field(default_factory=list)
    return_combined_text: bool = True
    debug: bool = False


class TextEvalResponse(BaseModel):
    complete: bool
    improve_message: Optional[str] = None
    combined_text: Optional[str] = None
    high_danger: bool = False
    question_relevant: bool = True  # NEW
    courtesy: bool = False  # NEW
    latency_ms: int


# Image
class ImageEvalJSONRequest(BaseModel):
    image_gcs_uri: Optional[str] = None


class ImageMetrics(BaseModel):
    blur_laplacian_var: float
    exposure_hist_entropy: float
    under_over_exposed_ratio: float
    noise_estimate_sigma: float
    compression_artifacts_score: float
    motion_blur_index: float
    skin_patch_detected: bool
    skin_area_ratio: float
    exif_orientation: int


class ImageEvalResponse(BaseModel):
    usable: bool
    improve_message: str
    metrics: ImageMetrics
    latency_ms: int
    source: str


# VL model request (image file in bucket)
class VLPredictRequestGCS(BaseModel):
    text_raw: Optional[str] = None
    image_gcs: Optional[str] = None


# VL model request (image encoded as string)
class VLPredictRequestBase64(BaseModel):
    text_raw: Optional[str] = None
    image_base64: Optional[str] = None


# VL model response (prediction)
class VLPredictResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Optional[dict] = None


# RAG / orchestrator interactions
class RAGRequest(BaseModel):
    question: str
    symptoms: Optional[str] = None
    conf: Optional[float] = None
    bug_class: Optional[str] = None


# class RAGResponse(BaseModel):
#     question: Optional[str] = None
#     prompt: Optional[str] = None
#     context: Optional[str] = None
#     bug_class: Optional[str] = None


class RAGModelPayload(BaseModel):
    question: Optional[str] = None
    prompt: Optional[str] = None
    context: Optional[str] = None
    bug_class: Optional[str] = None
    conf: Optional[float] = None


class RAGModelWrapper(BaseModel):
    status: str
    payload: RAGModelPayload
    latency_ms: Optional[int] = None


# Orchestrator evaluate request
class OrchestratorEvaluateRequest(BaseModel):
    image_gcs_uri: Optional[str] = None
    image_base64: Optional[str] = None
    user_text: Optional[str] = None
    overwrite_validation: bool = False
    first_call: bool = True
    history: List[str] = Field(default_factory=list)
    return_combined_text: bool = True
    debug: bool = False


class OrchestratorEvaluateResponse(BaseModel):
    ok: bool
    # For initial calls (prediction)
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    message: Optional[str] = None
    results: Optional[dict] = None
    image_issue: Optional[str] = None
    text_issue: Optional[str] = None
    error: Optional[str] = None
    # For followup calls (evaluation)
    eval: Optional[dict] = None
