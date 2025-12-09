#!/bin/bash
set -e

# Environment variables assumed to be set: GCS_BUCKET, BEST_MODEL_FILE, PULUMI_CONFIG_KEY

echo "💡 Starting VL model deployment check..."

# Download latest best_model.txt from GCS
echo "Downloading latest best_model.txt from GCS..."
gsutil cp gs://$GCS_BUCKET/$BEST_MODEL_FILE current_best_model.txt

# Read new label
NEW_LABEL=$(cat current_best_model.txt)
echo "New model label: $NEW_LABEL"

# Move to Pulumi deployment directory
cd /deploy_k8s

# Read current Pulumi config
CURRENT_LABEL=$(pulumi config get $PULUMI_CONFIG_KEY --stack dev || echo "none")
echo "Current Pulumi label: $CURRENT_LABEL"

# Compare and deploy if changed
if [ "$NEW_LABEL" != "$CURRENT_LABEL" ]; then
    echo "Model label changed — updating Pulumi config and deploying..."
    pulumi config set $PULUMI_CONFIG_KEY "$NEW_LABEL" --stack dev
    pulumi config set artifact_model_version "latest" --stack dev
    pulumi up --stack dev --yes
else
    echo "Model label unchanged — skipping deployment."
fi
