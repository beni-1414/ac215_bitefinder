#!/bin/bash

set -e # Exit on error

export SECRETS_DIR=../../../secrets/ # Path to secrets root directory
export DATA_DIR=../../data/ # Path to data root directory
export GCP_PROJECT="bitefinder-irith" # GCP project ID
export GOOGLE_APPLICATION_CREDENTIALS="../../../secrets/irith-service-account.json" # GCP service account JSON key
export IMAGE_NAME="vlmodel"

# Build Docker image
docker build -t $IMAGE_NAME -f Dockerfile .

# Run Docker container with mounted volumes for data and secrets
docker run --rm -ti \
    -v "$(pwd)"/$DATA_DIR:/app/data \
    -v "$(pwd)"/$SECRETS_DIR:/app/secrets \
    $IMAGE_NAME