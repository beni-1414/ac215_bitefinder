#!/bin/bash

# Package the vision-language model trainer for Google Cloud Vertex AI Custom Training
# This script creates a Python package that can be uploaded to GCS and used for serverless training

echo "📦 Packaging trainer for Google Cloud..."

# Source environment variables
source ../../env.dev

# Define package directory
PACKAGE_DIR="package/trainer"
PACKAGE_BASE_DIR="package"

# Create package directory if it doesn't exist
mkdir -p $PACKAGE_DIR

# Copy all necessary Python modules into the package
echo "📋 Copying modules into package..."
cp -v trainer.py $PACKAGE_DIR/trainer_module.py
cp -v model.py $PACKAGE_DIR/
cp -v dataset.py $PACKAGE_DIR/
cp -v utils_dataloader.py $PACKAGE_DIR/
cp -v utils_gcp.py $PACKAGE_DIR/
cp -v utils_save.py $PACKAGE_DIR/
cp -v utils_wandb.py $PACKAGE_DIR/

echo "✅ All modules copied to $PACKAGE_DIR/"

# Clean up any existing tar files
echo "🧹 Cleaning up old packages..."
rm -f trainer.tar trainer.tar.gz

# Create tar archive of the package
echo "📦 Creating tar archive..."
tar cvf trainer.tar $PACKAGE_BASE_DIR/

# Compress the archive
echo "🗜️  Compressing archive..."
gzip trainer.tar

# Upload to Google Cloud Storage
echo "☁️  Uploading to GCS..."
gsutil cp trainer.tar.gz $GCP_BUCKET_NAME/vlmodel_trainer.tar.gz

echo "✅ Package uploaded to $GCP_BUCKET_NAME/vlmodel_trainer.tar.gz"
echo ""
echo "🎯 To use this package with Vertex AI Custom Training, reference:"
echo "   $GCP_BUCKET_NAME/vlmodel_trainer.tar.gz"

# Remove copied files from the package directory (but keep the folder itself)
echo "🧽 Cleaning working directory..."
rm -f \
  "$PACKAGE_DIR/trainer_module.py" \
  "$PACKAGE_DIR/model.py" \
  "$PACKAGE_DIR/dataset.py" \
  "$PACKAGE_DIR/utils_dataloader.py" \
  "$PACKAGE_DIR/utils_gcp.py" \
  "$PACKAGE_DIR/utils_save.py" \
  "$PACKAGE_DIR/utils_wandb.py"

rm -f trainer.tar.gz