from google.cloud import secretmanager
import os
from google.auth import default


def get_secret(secret_id: str) -> str:
    """Retrieve a secret from Google Secret Manager."""
    try:
        credentials, _ = default()  # Use Application Default Credentials
        client = secretmanager.SecretManagerServiceClient(credentials=credentials)
        name = f"projects/{os.getenv('GCP_PROJECT')}/secrets/{secret_id}/versions/latest"
        response = client.access_secret_version(name=name)
        if response.payload.data:
            return response.payload.data.decode("UTF-8")
        return ""
    except Exception as e:
        print(f'⚠️ Failed to retrieve secret {secret_id}: {e}')
        return ""
