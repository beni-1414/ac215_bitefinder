from pydantic import BaseModel
import os


class Settings(BaseModel):
    GOOGLE_CLOUD_PROJECT: str | None = (
        os.getenv("GCP_PROJECT")  # your local choice
        or os.getenv("GOOGLE_CLOUD_PROJECT")  # set by Cloud Run / Cloud Functions
        or os.getenv("GCLOUD_PROJECT")  # used in some contexts
    )
    VERTEX_REGION: str = os.getenv("GCP_REGION", "us-central1")
    VERTEX_MODEL_NAME: str = os.getenv("VERTEX_MODEL_NAME", "gemini-2.5-flash")

    PORT: int = int(os.getenv("PORT", 8080))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
    ALLOW_ORIGINS: str = os.getenv("ALLOW_ORIGINS", "*")
    # Downstream service URLs (can be overridden via env)
    INPUT_EVAL_URL: str = os.getenv("INPUT_EVAL_URL", "http://input-evaluation:9000")
    VL_MODEL_URL: str = os.getenv("VL_MODEL_URL", "http://vl-model:9000")
    RAG_MODEL_URL: str = os.getenv("RAG_MODEL_URL", "http://rag-model:9000")


settings = Settings()
