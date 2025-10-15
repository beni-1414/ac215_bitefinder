from google.cloud import storage
import os

gcp_bucket_name = os.getenv('GCP_BUCKET_NAME')
gcp_project = os.getenv('GCP_PROJECT')

def upload_file_to_bucket(fp, storage_fp):
    if gcp_bucket_name and gcp_project:
        try:
            storage_client = storage.Client(project=gcp_project)
            bucket = storage_client.bucket(gcp_bucket_name)
            blob = bucket.blob(storage_fp)
            blob.upload_from_filename(fp)
            print(f'✅ Uploaded {fp} to GCP bucket {gcp_bucket_name}')
        except Exception as e:
            print(f'⚠️ Failed to upload to GCP: {e}')
    elif gcp_bucket_name is None:
        print(f'‼️ Skipping upload to GCP (misssing GCP_BUCKET_NAME)')
    else:
        print(f'‼️ Skipping upload to GCP (misssing GCP_PROJECT)')

def download_file_from_gcp(storage_fp, fp):
    if gcp_bucket_name and gcp_project:
        try:
            storage_client = storage.Client(project=gcp_project)
            bucket = storage_client.bucket(gcp_bucket_name)
            blob = bucket.blob(storage_fp)
            blob.download_to_filename(fp)
            print(f'✅ Downloaded {storage_fp} to GCP bucket {gcp_bucket_name}')
        except Exception as e:
            print(f'⚠️ Failed to download from GCP: {e}')
    elif gcp_bucket_name is None:
        print(f'‼️ Skipping download from GCP (misssing GCP_BUCKET_NAME)')
    else:
        print(f'‼️ Skipping download from GCP (misssing GCP_PROJECT)')

def download_directory_from_gcp(storage_p):
    if gcp_bucket_name and gcp_project:
        try:
            storage_client = storage.Client(project=gcp_project)
            bucket = storage_client.bucket(gcp_bucket_name)
            blobs = bucket.list_blobs(prefix=storage_p)
            for blob in blobs:
                if not blob.name.endswith('/'):
                    blob.download_to_filename(blob.name)
            print(f'✅ Downloaded {storage_p} to GCP bucket {gcp_bucket_name}')
        except Exception as e:
            print(f'⚠️ Failed to download from GCP: {e}')
    elif gcp_bucket_name is None:
        print(f'‼️ Skipping download from GCP (misssing GCP_BUCKET_NAME)')
    else:
        print(f'‼️ Skipping download from GCP (misssing GCP_PROJECT)')