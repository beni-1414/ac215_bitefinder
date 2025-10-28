#!/bin/bash

# Submit a Vertex AI training job using the JSON configuration

set -e

echo "🚀 Submitting Vertex AI training job..."

# Source environment variables
source ../../env.dev
export RUN_ID=labels_v2_$(date +%Y%m%d_%H%M%S)

# Authenticate with Google Cloud, comment this line if running inside GCP VM
# gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS

# Submit the job
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name="bitefinder-vlmodel-training-${RUN_ID}" \
  --python-package-uris="gs://bitefinder-data/vlmodel_trainer.tar.gz" \
  --worker-pool-spec=machine-type=e2-standard-4,replica-count=1,executor-image-uri=us-docker.pkg.dev/vertex-ai/training/pytorch-xla.2-4.py310:latest,python-module=trainer.task \
  --args="--model=clip","--epochs=10","--batch_size=32","--labels=texts_v2.json","--lr=0.0001","--gcp","--verbose","--run_id=${RUN_ID}"



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
