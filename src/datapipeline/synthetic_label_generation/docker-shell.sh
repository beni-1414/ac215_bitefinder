#!/bin/bash

set -e # Exit on error

export IMAGE_NAME="synthetic-label-generation"

# Build Docker image
docker build -t $IMAGE_NAME -f Dockerfile .

# Run Docker container with mounted volumes
docker run --rm --name $IMAGE_NAME -ti \
    -v "$(pwd)":/app \
    $IMAGE_NAME