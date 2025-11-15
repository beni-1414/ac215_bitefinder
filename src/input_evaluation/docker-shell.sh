#!/usr/bin/env bash
set -e

source ../../env.dev
IMAGE_NAME="bitefinder-input-evaluation"

# Create the network if we don't have it yet
docker network inspect bitefinder-network >/dev/null 2>&1 || docker network create bitefinder-network

# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME -f Dockerfile .

# Run the container
docker run --rm --name $IMAGE_NAME -ti \
  -v "$(pwd)"/api:/app/api \
  -v "$(pwd)"/$SECRETS_DIR:/secrets \
  -p 9000:9000 \
  -e GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS \
  -e DEV=1 \
  -e GCP_PROJECT=$GCP_PROJECT \
  -e GCS_BUCKET_URI=$GCS_BUCKET_URI \
  --network bitefinder-network \
  $IMAGE_NAME
