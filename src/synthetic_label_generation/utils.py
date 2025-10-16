import os
import json
import random
import time
from datetime import datetime
from typing import Dict, List

from openai import OpenAI
from google.cloud import storage, secretmanager


# ---------------------------
# Secret Management
# ---------------------------
def get_secret(secret_id: str) -> str:
    """Retrieve a secret from Google Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{os.getenv('GCP_PROJECT')}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(name=name)
    return response.payload.data.decode("UTF-8")


# ---------------------------
# OpenAI Client Initialization
# ---------------------------
def init_openai_client() -> OpenAI:
    """Initialize the OpenAI client using secret from Secret Manager."""
    api_key = get_secret("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    return OpenAI(api_key=api_key)


# ---------------------------
# Data Loading
# ---------------------------
def load_json_file(path: str) -> Dict:
    """Load a JSON file from a given path."""
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------
# Prompt + API Request
# ---------------------------
def generate_batch_labels(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float = 0.6
) -> Dict[str, List[str]]:
    """Send a batch prompt to OpenAI and return parsed JSON."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        response_format={"type": "json_object"}
    )

    output_text = response.choices[0].message.content.strip()
    return json.loads(output_text)


# ---------------------------
# Result Saving
# ---------------------------
def save_json(data: Dict, filename: str, output_dir: str = "output") -> str:
    """Save a JSON object to a file and return the file path."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, filename)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)
    return file_path


# ---------------------------
# GCS Upload
# ---------------------------
def upload_to_gcs(bucket_name: str, file_path: str, blob_name: str):
    """Upload a file to Google Cloud Storage under the given blob name."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(file_path)
    print(f"✅ Uploaded {file_path} to GCP bucket {bucket_name} as {blob_name}")


# ---------------------------
# Helper
# ---------------------------
def timestamp_suffix() -> str:
    """Return a timestamp suffix for filenames (e.g., 2025-10-16_14-30-00)."""
    return datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
