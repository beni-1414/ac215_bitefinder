from google.cloud import storage
import os

gcp_bucket_name = os.getenv('GCP_BUCKET_NAME')
gcp_project = os.getenv('GCP_PROJECT')

def upload_file_to_bucket(fp):
    if gcp_bucket_name and gcp_project:
        try:
            storage_client = storage.Client(project=gcp_project)
            bucket = storage_client.bucket(gcp_bucket_name)
            blob = bucket.blob(fp)
            blob.upload_from_filename(fp)
            print(f'✅ Uploaded {fp} to GCP bucket {gcp_bucket_name}')
        except Exception as e:
            print(f'⚠️ Failed to upload to GCP: {e}')
    elif gcp_bucket_name is None:
        print(f'‼️ Skipping upload to GCP (misssing GCP_BUCKET_NAME)')
    else:
        print(f'‼️ Skipping upload to GCP (misssing GCP_PROJECT)')

def download_file_from_gcp(storage_fp):
    if gcp_bucket_name and gcp_project:
        try:
            storage_client = storage.Client(project=gcp_project)
            bucket = storage_client.bucket(gcp_bucket_name)
            blob = bucket.blob(storage_fp)
            blob.download_to_filename(blob.name)
            print(f'✅ Downloaded {storage_fp} from GCP bucket {gcp_bucket_name}')
        except Exception as e:
            print(f'⚠️ Failed to download from GCP: {e}')
    elif gcp_bucket_name is None:
        print(f'‼️ Skipping download from GCP (misssing GCP_BUCKET_NAME)')
    else:
        print(f'‼️ Skipping download from GCP (misssing GCP_PROJECT)')

def download_directory_from_gcp(storage_dir, local_dir='/app/'):
    if gcp_bucket_name and gcp_project:
        try:
            storage_client = storage.Client(project=gcp_project)
            bucket = storage_client.bucket(gcp_bucket_name)
            blobs = bucket.list_blobs(prefix=storage_dir)
            for blob in blobs:
                if not blob.name.endswith('/'):
                    local_path = os.path.join(local_dir, blob.name)
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    blob.download_to_filename(local_path)
            print(f'✅ Downloaded {storage_dir} from GCP bucket {gcp_bucket_name}')
        except Exception as e:
            print(f'⚠️ Failed to download from GCP: {e}')
    elif gcp_bucket_name is None:
        print(f'‼️ Skipping download from GCP (misssing GCP_BUCKET_NAME)')
    else:
        print(f'‼️ Skipping download from GCP (misssing GCP_PROJECT)')