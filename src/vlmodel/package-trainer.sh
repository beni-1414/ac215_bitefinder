#!/bin/bash

# Package the vision-language model trainer for Google Cloud Vertex AI Custom Training
# This script creates a Python package that can be uploaded to GCS and used for serverless training

echo "📦 Packaging trainer for Google Cloud..."

# Source environment variables
source ../../env.dev

# Define package directory
PACKAGE_BASE_DIR="package"

# Clean up any existing tar files
echo "🧹 Cleaning up old packages..."
rm -f trainer.tar trainer.tar.gz

# Create tar archive of the package
echo "📦 Creating tar archive..."
tar cvf trainer.tar -C $PACKAGE_BASE_DIR . # So setup.py is at the root of the archive

# Compress the archive
echo "🗜️  Compressing archive..."
gzip trainer.tar

# Upload to Google Cloud Storage
echo "☁️  Uploading to GCS..."
gsutil cp trainer.tar.gz $GCS_BUCKET_URI/vlmodel_trainer.tar.gz

echo "✅ Package uploaded to $GCS_BUCKET_URI/vlmodel_trainer.tar.gz"
echo ""
echo "🎯 To use this package with Vertex AI Custom Training, reference:"
echo "   $GCS_BUCKET_URI/vlmodel_trainer.tar.gz"

rm -f trainer.tar.gz