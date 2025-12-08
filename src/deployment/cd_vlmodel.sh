#!/bin/bash
set -e

cd /app

echo "Logging into Pulumi using GCS backend..."
pulumi login gs://bitefinder-474614-pulumi-state-bucket

echo "Reading current Pulumi config..."
CURRENT_LABEL=$(pulumi config get artifact_model_label --stack dev || echo "none")
echo "Current Pulumi label = $CURRENT_LABEL"

echo "Reading new best model..."
NEW_LABEL=$(cat /workspace/current_best_model.txt)
echo "New best_model.txt   = $NEW_LABEL"

if [ "$NEW_LABEL" = "$CURRENT_LABEL" ]; then
  echo "Model unchanged — exiting."
  exit 0
fi

echo "Updating Pulumi config..."
pulumi config set artifact_model_label "$NEW_LABEL" --stack dev
pulumi config set artifact_model_version "latest" --stack dev

echo "Running Pulumi up..."
pulumi up --stack dev --yes

echo "Deployment complete."
