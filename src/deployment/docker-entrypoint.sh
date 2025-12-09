#!/bin/bash

VENV_PATH="${VIRTUAL_ENV:-/home/app/.venv}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
  # Ensure Pulumi uses the project virtualenv
  source "${VENV_PATH}/bin/activate"
  export PULUMI_PYTHON_CMD="${VENV_PATH}/bin/python"
  echo "Using virtual environment at ${VENV_PATH}"
else
  echo "WARNING: Virtual environment not found at ${VENV_PATH}; falling back to system Python."
fi

echo "Container is running!!!"
echo "Architecture: $(uname -m)"
echo "Python version: $(python --version)"
echo "UV version: $(uv --version)"
echo "Python executable: $(which python)"

# Authenticate gcloud using service account
gcloud auth activate-service-account --key-file $GOOGLE_APPLICATION_CREDENTIALS
gcloud config set project $GCP_PROJECT
# login to artifact-registry
gcloud auth configure-docker us-east1-docker.pkg.dev --quiet
# Check if the bucket exists
if ! gsutil ls -b $PULUMI_BUCKET >/dev/null 2>&1; then
    echo "Bucket does not exist. Creating..."
    gsutil mb -p $GCP_PROJECT $PULUMI_BUCKET
else
    echo "Bucket already exists. Skipping creation."
fi

echo "Logging into Pulumi using GCS bucket: $PULUMI_BUCKET"
pulumi login $PULUMI_BUCKET

# List available stacks
echo "Available Pulumi stacks in GCS:"
gsutil ls $PULUMI_BUCKET/.pulumi/stacks/  || echo "No stacks found."

args="$@"
echo $args

if [[ -z ${args} ]];
then
    /bin/bash
else
  /bin/bash $args
fi
