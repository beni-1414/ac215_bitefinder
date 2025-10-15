import json
from google.cloud import storage
import os

# 📝 Set your bucket name via env var (best practice)
BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "bitefinder-data")

# 📁 Folder inside the bucket where your JSON files are stored
PREFIX = "synthetic-labels"

# 📂 File names
locations_file = f"{PREFIX}/locations.json"
symptoms_file = f"{PREFIX}/symptoms.json"

# 🧠 Initialize the client (no creds needed if running on GCP VM with service account)
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

def read_json_from_bucket(blob_name: str):
    blob = bucket.blob(blob_name)
    data_bytes = blob.download_as_bytes()
    return json.loads(data_bytes)

# 🌐 Load the files from GCS
locations_data = read_json_from_bucket(locations_file)
symptoms_data = read_json_from_bucket(symptoms_file)

print(locations_data)

print("✅ Loaded JSON files from GCS")