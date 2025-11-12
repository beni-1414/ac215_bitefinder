#!/bin/bash

set -e # Exit on error

source ../../env.dev # Read the settings file
export IMAGE_NAME="vlmodel"

# Build Docker image
docker build -t $IMAGE_NAME -f Dockerfile .

# Run Docker container
docker run --rm --name $IMAGE_NAME -ti \
    -p 8080:8080 \
    -v "$(pwd)":/app \
    -e GCP_BUCKET_NAME=$GCP_BUCKET_NAME \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e WANDB_PROJECT=$WANDB_PROJECT \
    -e WANDB_TEAM=$WANDB_TEAM \
    $IMAGE_NAME