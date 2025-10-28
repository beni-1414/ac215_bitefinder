#!/bin/bash

# Submit a Vertex AI training job using the JSON configuration

set -e

echo "🚀 Submitting Vertex AI training job..."

# Source environment variables
source ../../env.dev

# Authenticate with Google Cloud, comment this line if running inside GCP VM
gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS

# Submit the job
gcloud ai custom-jobs create \
    --region=us-central1 \
    --config=vertex-ai-training-job.json \
    --project=${GCP_PROJECT}

echo ""
echo "✅ Training job submitted successfully!"
echo ""
echo "📊 Monitor your job:"
echo "   Console: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${GCP_PROJECT}"
echo "   Logs: https://console.cloud.google.com/logs/query?project=${GCP_PROJECT}"
echo "   W&B: https://wandb.ai/bitefinder/bitefinder-vl"
echo ""
echo "📝 To list all jobs:"
echo "   gcloud ai custom-jobs list --region=us-central1 --project=${GCP_PROJECT}"
echo ""
echo "🔍 To get job details:"
echo "   gcloud ai custom-jobs describe <JOB_ID> --region=us-central1 --project=${GCP_PROJECT}"
