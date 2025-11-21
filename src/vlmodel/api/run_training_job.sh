#!/bin/bash

# Submit a Vertex AI training job using the JSON configuration

set -e

echo "🚀 Submitting Vertex AI training job..."

# Source environment variables
source ../../../env.dev

export RUN_ID=run_$(date +%Y%m%d_%H%M%S)
export REPLICA_COUNT=1

# Authenticate with Google Cloud, comment this line if running inside GCP VM
gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS

# Set up W&B sweep if sweep config file is provided
SWEEP_CONFIG=$1
if [ -n "$SWEEP_CONFIG" ]; then
    echo "🌀 Creating W&B sweep from sweep.yaml..."
    SWEEP_ID=$(wandb sweep --project "$WANDB_PROJECT" --entity "$WANDB_TEAM" "$SWEEP_CONFIG" 2>&1 | awk '/Created sweep with ID:/ {print $NF}')
    if [ -z "$SWEEP_ID" ]; then
        echo "Error: Could not extract sweep ID"
        exit 1
    fi
    export SWEEP_ID
    export REPLICA_COUNT=4 # Number of parallel workers
    echo "✅ Sweep created: $SWEEP_ID"
fi

# Produce a resolved config with actual values baked in
envsubst < job_config.yaml > job_config_resolved.yaml

gcloud ai custom-jobs create \
  --region=$GCP_REGION \
  --display-name="bitefinder-vlmodel-training-${RUN_ID}" \
  --project=$GCP_PROJECT \
  --config=job_config_resolved.yaml \
  --python-package-uris=$PYTHON_PACKAGE_URI \
  --service-account=$SERVICE_ACCOUNT_EMAIL

echo ""
echo "✅ Training job submitted successfully!"
echo ""
echo "📊 Monitor your job:"
echo "   Console: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${GCP_PROJECT}"
echo "   W&B: https://wandb.ai/bitefinder/bitefinder-vl"
echo ""

rm -f job_config_resolved.yaml
