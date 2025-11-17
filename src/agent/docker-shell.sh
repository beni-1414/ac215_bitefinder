#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Define some environment variables
export IMAGE_NAME="agent-api-service"
export BASE_DIR=$(pwd)
export SECRETS_DIR=$(pwd)/../../../secrets/
export PERSISTENT_DIR=$(pwd)/../../../persistent-folder/
export GCP_PROJECT="mineral-style-471117-k1" #"apcomp215"
#export GCS_BUCKET_NAME="cheese-app-models"

export GOOGLE_APPLICATION_CREDENTIALS="/secrets/llm-service-account.json"
export PINECONE_API_KEY="$(cat ../../../secrets/pineconeAPI.txt)"
export PINECONE_INDEX="bugbite-rag"
export PINECONE_CLOUD="aws"
export PINECONE_REGION="us-east-1"



# Create the network if we don't have it yet
docker network inspect llm-rag-network >/dev/null 2>&1 || docker network create llm-rag-network


# Build the image based on the Dockerfile
docker build -t $IMAGE_NAME -f Dockerfile .

# Run the container
docker run --rm --name $IMAGE_NAME -ti \
-v "$BASE_DIR":/app \
-v "$SECRETS_DIR":/secrets \
-v "$PERSISTENT_DIR":/persistent \
-p 9000:9000 \
-e DEV=1 \
-e GOOGLE_APPLICATION_CREDENTIALS=/secrets/llm-service-account.json \
-e GCP_PROJECT=$GCP_PROJECT \
-e PINECONE_API_KEY=$PINECONE_API_KEY \
-e PINECONE_INDEX=$PINECONE_INDEX \
--network llm-rag-network \
$IMAGE_NAME

#-e GCS_BUCKET_NAME=$GCS_BUCKET_NAME \
#-e CHROMADB_HOST=$CHROMADB_HOST \
#-e CHROMADB_PORT=$CHROMADB_PORT \