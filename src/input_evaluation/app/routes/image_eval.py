from __future__ import annotations
import time
from fastapi import APIRouter, HTTPException
from app.schemas import ImageEvalJSONRequest, ImageEvalResponse, ImageMetrics
from app.services.gcs_io import read_bytes_gcs
from app.services.image_quality import compute_metrics, decide

router = APIRouter(prefix="/v1/evaluate", tags=["evaluate"])


@router.post("/image", response_model=ImageEvalResponse)
async def evaluate_image(json_req: ImageEvalJSONRequest) -> ImageEvalResponse:
    start = time.perf_counter()

    if not json_req.image_gcs_uri or not json_req.image_gcs_uri.startswith("gs://"):
        # Optional: let Pydantic validate this instead (see below)
        raise HTTPException(status_code=422, detail="image_gcs_uri must be a gs:// URI")

    img_bytes = read_bytes_gcs(json_req.image_gcs_uri)
    m = compute_metrics(img_bytes)
    usable, msg, _ = decide(m)

    latency_ms = int((time.perf_counter() - start) * 1000)
    return ImageEvalResponse(
        usable=usable,
        improve_message=msg,
        metrics=ImageMetrics(**m.__dict__),
        latency_ms=latency_ms,
        source="gcs",
    )
