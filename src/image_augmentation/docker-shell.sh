#!/bin/bash
set -e
source ../../../env.dev
export IMAGE_NAME="image-augmentation"

docker build -t $IMAGE_NAME -f Dockerfile .

docker run --rm --name $IMAGE_NAME -ti \
    -v "$(pwd)":/app \
    -v "$(pwd)"/$DATA_DIR:/app/data \
    -v "$HOME/ac215/secrets":/secrets \
    -e GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS \
    -e GCP_PROJECT=$GCP_PROJECT \
    -e GCP_BUCKET_NAME=$GCP_BUCKET_NAME \
    $IMAGE_NAME