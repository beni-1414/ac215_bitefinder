#!/bin/bash

set -e # Exit on error

source ../../env.dev # Read the settings file
export IMAGE_NAME="bitefinder-vlmodel"

# Create the network if we don't have it yet
docker network inspect bitefinder-network >/dev/null 2>&1 || docker network create bitefinder-network

# Build Docker image
docker build -t $IMAGE_NAME -f Dockerfile .

# Run Docker container
docker run --rm --name $IMAGE_NAME -ti \
    -p 9000:9000 \
    -v "$(pwd)":/app \
    -v "$(pwd)/$MODEL_CACHE_DIR":/$MODEL_CACHE_DIR \
    -v "$(pwd)"/$SECRETS_DIR:/secrets \
    -e GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS \
    -e GCP_BUCKET_NAME=$GCP_BUCKET_NAME \
    -e SECRET_MANAGER_PROJECT=$SECRET_MANAGER_PROJECT \
    -e WANDB_PROJECT=$WANDB_PROJECT \
    -e WANDB_TEAM=$WANDB_TEAM \
    -e MODEL_CACHE_DIR=$MODEL_CACHE_DIR \
    --network bitefinder-network \
    $IMAGE_NAME
