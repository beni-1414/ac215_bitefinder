#!/bin/bash

# Submit a Vertex AI training job using the JSON configuration

set -e

echo "🚀 Submitting Vertex AI training job..."

# Source environment variables
source ../../env.dev
export RUN_ID=labels_v2_$(date +%Y%m%d_%H%M%S)

# Authenticate with Google Cloud, comment this line if running inside GCP VM
gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS

# Produce a resolved config with actual values baked in
envsubst < job_config.yaml > job_config_resolved.yaml

gcloud ai custom-jobs create \
  --region=$GCP_REGION \
  --display-name="bitefinder-vlmodel-training-${RUN_ID}" \
  --project=$GCP_PROJECT \
  --config=job_config_resolved.yaml \
  --python-package-uris=$PYTHON_PACKAGE_URI \
  --service-account=irith-service-account@bitefinder-irith.iam.gserviceaccount.com

echo ""
echo "✅ Training job submitted successfully!"
echo ""
echo "📊 Monitor your job:"
echo "   Console: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${GCP_PROJECT}"
echo "   W&B: https://wandb.ai/bitefinder/bitefinder-vl"
echo ""
echo "🔍 To get job details:"
echo "   gcloud ai custom-jobs describe <JOB_ID> --region=${GCP_REGION} --project=${GCP_PROJECT}"

rm -f job_config_resolved.yaml