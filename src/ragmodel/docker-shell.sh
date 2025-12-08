#!/usr/bin/env bash
set -e

source ../../env_TEMPLATE.dev #source ../../env.dev
IMAGE_NAME="bitefinder-rag"

# Create Docker network if missing
docker network inspect bitefinder-network >/dev/null 2>&1 || docker network create bitefinder-network

# Build the RAG image
docker build -t $IMAGE_NAME -f Dockerfile .

# Run container with hot reload + developer mode
docker run --rm --name $IMAGE_NAME -ti \
  -v "$(pwd)"/api:/app/api \
  -v "$(pwd)"/$SECRETS_DIR:/secrets \
  -p 9000:9000 \
  -e GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS \
  -e DEV=1 \
  -e GCP_PROJECT=$GCP_PROJECT \
  -e PINECONE_API_KEY="$PINECONE_API_KEY" \
  -e PINECONE_INDEX="$PINECONE_INDEX" \
  -e PINECONE_CLOUD="$PINECONE_CLOUD" \
  -e PINECONE_REGION="$PINECONE_REGION" \
  --network bitefinder-network \
  $IMAGE_NAME
