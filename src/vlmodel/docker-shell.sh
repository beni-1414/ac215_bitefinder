#!/bin/bash

set -e # Exit on error

source ../../env.dev # Read the settings file
export IMAGE_NAME="vlmodel"

# Build Docker image
docker build -t $IMAGE_NAME -f Dockerfile .

# Run Docker container with mounted volumes for data and secrets
docker run --rm --name $IMAGE_NAME -ti \
    -v "$(pwd)":/app \
    -v "$(pwd)"/$DATA_DIR:/app/data \
    -v "$(pwd)"/$SECRETS_DIR:/app/secrets \
    $IMAGE_NAME