# Serverless Training with Google Cloud Vertex AI

This guide explains how to package and run your vision-language model training on Google Cloud Vertex AI for serverless training.

## Overview

The training code has been structured as a Python package that can be uploaded to Google Cloud Storage and executed on Vertex AI Custom Training jobs. This enables serverless, scalable training without managing infrastructure.

## Package Structure

```
trainer/
├── __init__.py              # Package initialization
├── task.py                  # Entry point for training job
├── trainer_module.py        # Main Trainer class
├── model.py                 # Model definitions (CLIP, ViLT)
├── dataset.py               # Dataset class
├── utils_dataloader.py      # Data loading utilities
├── utils_gcp.py             # GCP utilities
├── utils_save.py            # Model saving utilities
└── utils_wandb.py           # Weights & Biases configuration
```

## Prerequisites

1. **Google Cloud SDK** installed and configured
2. **Environment variables** set in `env.dev`:
   - `GCP_PROJECT`: Your Google Cloud project ID
   - `GCP_BUCKET_NAME`: Your GCS bucket name (e.g., `gs://bitefinder-data`)
3. **Data uploaded to GCS**: Your training data should be in the GCS bucket at `data/`
4. **Service Account**: A properly configured service account with necessary permissions (see Step 0 below)


### WARNING: Give your default compute account access to the secrets

If you prefer to use the default Vertex AI service account, grant it Secret Manager access:

```bash
# Find your project number
PROJECT_NUMBER=$(gcloud projects describe bitefinder-474614 --format="value(projectNumber)")

# Grant Secret Manager access
gcloud secrets add-iam-policy-binding WANDB_API_KEY \
    --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor" \
    --project=bitefinder-474614
```

You can also do it from the service accounts display in GCP.

## Step 1: Create the Package

Run the packaging script from the `vlmodel` directory:

```bash
bash package-trainer.sh
```

This script will:
1. Copy all necessary Python modules into the `trainer/` directory
2. Create a compressed tar archive (`trainer.tar.gz`)
3. Upload the package to your GCS bucket at `{bucket}/vlmodel_trainer.tar.gz`

## Step 2: Submit a Vertex AI Training Job

### Option A: Using JSON Configuration (Recommended)

The easiest way is to use the pre-configured JSON file:

```bash
bash submit-training-job.sh
```

Or manually:
```bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --config=vertex-ai-training-job.json \
  --project=bitefinder-474614
```

### Option B: Using Python SDK:

```python
from google.cloud import aiplatform

aiplatform.init(project='bitefinder-474614', location='us-central1')

job = aiplatform.CustomPythonPackageTrainingJob(
    display_name='bitefinder-vlmodel-training',
    python_package_gcs_uri='gs://bitefinder-data/vlmodel_trainer.tar.gz',
    python_module_name='trainer.task',
    container_uri='us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13:latest',
)

job.run(
    args=[
        '--model=clip',
        '--epochs=10',
        '--batch_size=32',
        '--lr=0.0001',
        '--gcp',
        '--verbose'
    ],
    replica_count=1,
    machine_type='n1-standard-4',
    accelerator_type='NVIDIA_TESLA_T4',
    accelerator_count=1,
    service_account='bitefinder-training@bitefinder-474614.iam.gserviceaccount.com',
    environment_variables={
        'GCP_PROJECT': 'bitefinder-474614',
        'GCP_BUCKET_NAME': 'gs://bitefinder-data'
    }
)
```

### Customizing vertex-ai-training-job.json

You can edit `vertex-ai-training-job.json` to customize:
- Machine type and GPU configuration
- Training arguments (epochs, batch size, learning rate, etc.)
- Service account
- Environment variables

```json
{
  "displayName": "bitefinder-vlmodel-training",
  "jobSpec": {
    "workerPoolSpecs": [{
      "machineSpec": {
        "machineType": "n1-standard-4",
        "acceleratorType": "NVIDIA_TESLA_T4",
        "acceleratorCount": 1
      },
      "replicaCount": 1,
      "pythonPackageSpec": {
        "executorImageUri": "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13:latest",
        "packageUris": ["gs://bitefinder-data/vlmodel_trainer.tar.gz"],
        "pythonModule": "trainer.task",
        "args": ["--model=clip", "--epochs=10", "--batch_size=32", "--lr=0.0001", "--gcp", "--verbose"],
        "env": [
          {"name": "GCP_PROJECT", "value": "bitefinder-474614"},
          {"name": "GCP_BUCKET_NAME", "value": "gs://bitefinder-data"}
        ]
      }
    }],
    "serviceAccount": "bitefinder-training@bitefinder-474614.iam.gserviceaccount.com"
  }
}
```

## Training Arguments

The `task.py` script accepts the following arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `clip` | Model type: `clip` or `vilt` |
| `--epochs` | int | `10` | Number of training epochs |
| `--batch_size` | int | `64` | Batch size for training |
| `--lr` | float | `1e-4` | Learning rate |
| `--device` | str | auto | Device (cuda/cpu) - auto-detected if not specified |
| `--labels` | str | `texts_v1.json` | Path to labels JSON file |
| `--data_root_dir` | str | `data/` | Root directory for training data |
| `--output_dir` | str | `trained_model` | Directory to save trained model |
| `--seed` | int | `42` | Random seed for reproducibility |
| `--verbose` | flag | False | Enable verbose output |
| `--gcp` | flag | False | Download data from GCP bucket |

## Environment Variables for Training Job

Your training job needs these environment variables (set in `vertex-ai-training-job.json`):

```bash
GCP_PROJECT=bitefinder-474614
GCP_BUCKET_NAME=gs://bitefinder-data
```

**Note**: `WANDB_API_KEY` is retrieved from Secret Manager (not passed as env var) for security.

## Monitoring Training

1. **Vertex AI Console**: Monitor job status, logs, and resource usage
2. **Weights & Biases**: View training metrics, loss curves, and accuracy
3. **Cloud Logging**: Access detailed logs from the training job

## Output

The trained model will be saved to the specified `output_dir` with:
- `config.pt`: Model configuration and label mappings
- Pre-trained model weights
- Fine-tuned classifier weights (`classifier.pt`)

If `--gcp` flag is used with output, the model can be automatically uploaded to your GCS bucket.

## Troubleshooting

### Secret Manager access errors
- **Error**: `Failed to retrieve secret WANDB_API_KEY`
- **Cause**: Service account doesn't have Secret Manager access
- **Solution**: Run `setup-service-account.ps1` or manually grant `roles/secretmanager.secretAccessor`
- **Verify**: Check IAM policy with `gcloud secrets get-iam-policy WANDB_API_KEY`

### Data loading errors
- Ensure data is uploaded to GCS at the expected path
- Use `--gcp` flag to download data from GCS
- Check that `GCP_BUCKET_NAME` environment variable is set correctly
