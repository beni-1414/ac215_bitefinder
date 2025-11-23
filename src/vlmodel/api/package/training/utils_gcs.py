import subprocess
import os


def download_from_gcs(bucket_dir):
    gcs_dir = os.getenv('GCS_BUCKET_URI') + '/' + bucket_dir
    print(f'☁️ Downloading data from {gcs_dir}...')
    subprocess.run(['gsutil', '-q', '-m', 'cp', '-r', gcs_dir, '.'], check=True)
    print('✅ Download complete!')
