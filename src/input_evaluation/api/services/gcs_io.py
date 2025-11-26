from __future__ import annotations
from typing import Tuple
from urllib.parse import urlparse
from google.cloud import storage
from fastapi import HTTPException, status

_client: storage.Client | None = None


def get_client() -> storage.Client:
    global _client
    if _client is None:
        _client = storage.Client()
    return _client


def parse_gs_uri(uri: str) -> Tuple[str, str]:
    if not uri.startswith("gs://"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="GCS URI must start with gs://")
    p = urlparse(uri)
    bucket = p.netloc
    blob = p.path.lstrip("/")
    return bucket, blob


def read_bytes_gcs(uri: str) -> bytes:
    bucket_name, blob_name = parse_gs_uri(uri)
    client = get_client()
    blob = client.bucket(bucket_name).blob(blob_name)
    return blob.download_as_bytes()
