# Serverless Training – Quick Start

## Prerequisites

* **OS:** Linux or macOS
  *(Windows isn’t supported by these scripts, use WSL Ubuntu, a Linux VM or macOS.)*
* **Google Cloud CLI** installed and authenticated.
* Access to the GCP **project**, **GCS bucket**, and **Vertex AI**.

## One-time setup checklist

* [ ] **Adapt `env_TEMPLATE.dev`** to your project (IDs, regions, bucket URIs, package URIs, etc.), and name it `env.dev`. If you already have an `env.dev`, ensure it’s up to date.
* [ ] **Service Account (SA):** use your own SA and update the scripts.
  Grant it:

  * `roles/iam.serviceAccountUser` (Service Account User)
  * `roles/aiplatform.admin` (Vertex AI Admin)
  * `roles/ml.admin` (AI Platform Admin – legacy, if you still use it)
  * `roles/logging.logWriter` (Logs Writer)
  * Access to your **GCS bucket** (e.g., `roles/storage.objectAdmin`).
    Also ensure **@Beni** has granted you bucket access.
* [ ] **Secrets Manager:** create a secret for your W&B key.

  * Put your **WANDB API key** there (name it `WANDB_API_KEY`).
  * Grant your **SA** permission to read it (`roles/secretmanager.secretAccessor`).

* IF YOU DO THIS, comment out the WANDBKEY in the yaml config to avoid errors!! Ideally remove alltogether when @Irith can test it works for him.

## Configure the job

Edit **`job_config.yaml`**:

* **Machine / accelerator:** set CPU or GPU you want:

  ```yaml
  machineSpec:
    machineType: e2-standard-4
    # accelerator_type: NVIDIA_TESLA_T4
    # accelerator_count: 1
  pythonPackageSpec:
    executorImageUri: us-docker.pkg.dev/vertex-ai/training/pytorch-xla.2-4.py310:latest  # CPU/TPU-XLA
    # executorImageUri: us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-4.py310:latest  # GPU
  ```
* **Training args:** adjust model/epochs/batch size/device:

  ```yaml
  args:
    - --model=vilt
    - --epochs=20
    - --batch_size=32
    - --device=cpu   # change to cuda for GPU images
    - --run_id=${RUN_ID}
    - --save
  ```
* **Keep the date tag** in RUN_ID (scripts add it automatically).

## Packaging (only if code changed)

If you modified Python code in the `package/` folder:

```bash
bash package_training.sh
```

This creates and uploads `vlmodel_training.tar.gz` to the GCS bucket.

## Submit the job

```bash
bash run_training.sh
```

Notes:

* The script sources `../../env.dev`, resolves `job_config.yaml` via `envsubst`, and sets `RUN_ID=labels_v2_<YYYYMMDD_HHMMSS>`.
* If running **outside** a GCP VM, the script uses:

  ```bash
  gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS
  ```

  Make sure the secrets folder is set up as usual outside the repo.

---

## Monitor progress

* **Vertex AI Console:** Jobs → Custom Jobs (link is printed after submit)
* **W&B:** Your `WANDB_TEAM` / `WANDB_PROJECT` (link printed after submit)
* **Logs:** Cloud Logging (also linked from the job page)