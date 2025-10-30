#!/bin/bash

set -e # Exit on error

source ../../env.dev # Read the settings file
export IMAGE_NAME="vlmodel"

# Build Docker image
docker build -t $IMAGE_NAME -f Dockerfile .

# Run Docker container with mounted volumes for data and secrets
docker run --rm --name $IMAGE_NAME -ti \
    -v "$(pwd)":/app \
    -v "$(pwd)"/$SECRETS_DIR:/secrets \
    -e GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS \
    -e GCP_PROJECT=$GCP_PROJECT \
    -e GCP_BUCKET_NAME=$GCP_BUCKET_NAME \
    -e GCS_BUCKET_URI=$GCS_BUCKET_URI \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    $IMAGE_NAME
    # -v "$(pwd)"/$DATA_DIR:/app/data \
