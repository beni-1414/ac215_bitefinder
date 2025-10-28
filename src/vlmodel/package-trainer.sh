#!/bin/bash

# Package the vision-language model trainer for Google Cloud Vertex AI Custom Training
# This script creates a Python package that can be uploaded to GCS and used for serverless training

echo "📦 Packaging trainer for Google Cloud..."

# Source environment variables
source ../../env.dev

# Define package directory
PACKAGE_DIR="trainer"

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

# Ensure __init__.py and task.py exist (they should already be there)
if [ ! -f "$PACKAGE_DIR/__init__.py" ]; then
    echo "# Package initialization for trainer module" > $PACKAGE_DIR/__init__.py
fi

# Verify task.py exists
if [ ! -f "$PACKAGE_DIR/task.py" ]; then
    echo "❌ Error: task.py not found in $PACKAGE_DIR/"
    echo "Please ensure task.py has been created in the trainer directory"
    exit 1
fi

echo "✅ All modules copied to $PACKAGE_DIR/"

# Clean up any existing tar files
echo "🧹 Cleaning up old packages..."
rm -f trainer.tar trainer.tar.gz

# Create tar archive of the package
echo "📦 Creating tar archive..."
tar cvf trainer.tar $PACKAGE_DIR/

# Compress the archive
echo "🗜️  Compressing archive..."
gzip trainer.tar

# Upload to Google Cloud Storage
echo "☁️  Uploading to GCS..."
gsutil cp trainer.tar.gz $GCP_BUCKET_NAME/vlmodel_trainer.tar.gz

echo "✅ Package uploaded to $GCP_BUCKET_NAME/vlmodel_trainer.tar.gz"
echo ""
echo "🎯 To use this package with Vertex AI Custom Training, reference:"
echo "   gs://${GCP_BUCKET_NAME#gs://}/vlmodel_trainer.tar.gz"