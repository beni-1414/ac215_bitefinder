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
