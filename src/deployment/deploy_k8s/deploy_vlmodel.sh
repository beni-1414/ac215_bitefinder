#!/bin/bash
set -e

# Set environment variables
export GCP_PROJECT="bitefinder-474614"
export GCP_REGION="us-central1"
export GCS_BUCKET="bitefinder-data"

echo "💡 Starting VL model deployment check..."

# Download latest best_model.txt from GCS
echo "Downloading latest best_model.txt from GCS..."
gsutil cp gs://$GCS_BUCKET/best_model.txt current_best_model.txt

# Read new label
NEW_LABEL=$(cat current_best_model.txt)
echo "New model label: $NEW_LABEL"

# Read current Pulumi config
CURRENT_LABEL=$(pulumi config get artifact_model_label --stack dev || echo "none")
echo "Current Pulumi label: $CURRENT_LABEL"

# Compare and deploy if changed
if [ "$NEW_LABEL" != "$CURRENT_LABEL" ]; then
    gcloud container clusters get-credentials bitefinder-cluster --region $GCP_REGION --project $GCP_PROJECT

    echo "Model label changed — updating Pulumi config and deploying..."
    pulumi config set artifact_model_label "$NEW_LABEL" --stack dev

    echo "Deleting existing vlmodel pod..."
    kubectl delete deployment vlmodel -n bitefinder-namespace

    echo "Refreshing Pulumi state..."
    pulumi refresh --stack dev --yes

    echo "Deploying new vlmodel pod..."
    pulumi up --stack dev --yes
else
    echo "Model label unchanged — skipping deployment."
fi

rm -f current_best_model.txt
