import os
from pydantic import BaseModel, Field


class Thresholds(BaseModel):
    MIN_LAPLACIAN_VAR: float = Field(45.0)
    MIN_EXPOSURE_ENTROPY: float = Field(3.0)
    MAX_EXPOSURE_CLIP_RATIO: float = Field(0.30)
    MAX_NOISE_SIGMA: float = Field(30.0)
    MAX_BLOCKINESS: float = Field(0.15)
    MAX_MOTION_BLUR: float = Field(0.60)
    MIN_SKIN_AREA_RATIO: float = Field(0.15)


class Settings(BaseModel):
    ALLOW_ORIGINS: str = os.getenv("ALLOW_ORIGINS", "*")

    GOOGLE_CLOUD_PROJECT: str | None = (
        os.getenv("GCP_PROJECT")  # your local choice
        or os.getenv("GOOGLE_CLOUD_PROJECT")  # set by Cloud Run / Cloud Functions
        or os.getenv("GCLOUD_PROJECT")  # used in some contexts
    )
    VERTEX_REGION: str = os.getenv("GCP_REGION", "us-central1")
    VERTEX_MODEL_NAME: str = os.getenv("VERTEX_MODEL_NAME", "gemini-2.5-flash")

    THRESHOLDS: Thresholds = Thresholds(
        MIN_LAPLACIAN_VAR=float(os.getenv("MIN_LAPLACIAN_VAR", 45)),
        MIN_EXPOSURE_ENTROPY=float(os.getenv("MIN_EXPOSURE_ENTROPY", 3.0)),
        MAX_EXPOSURE_CLIP_RATIO=float(os.getenv("MAX_EXPOSURE_CLIP_RATIO", 0.30)),
        MAX_NOISE_SIGMA=float(os.getenv("MAX_NOISE_SIGMA", 30)),
        MAX_BLOCKINESS=float(os.getenv("MAX_BLOCKINESS", 0.15)),
        MAX_MOTION_BLUR=float(os.getenv("MAX_MOTION_BLUR", 0.60)),
        MIN_SKIN_AREA_RATIO=float(os.getenv("MIN_SKIN_AREA_RATIO", 0.15)),
    )


settings = Settings()
