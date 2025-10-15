# VS Code ↔ GCP VM & Run Container — Step‑by‑Step Guide

This guide gives you two practical workflows:

1. **Develop in a GCP VM from VS Code** (Remote‑SSH)
2. **Spin up a VM just to run a container image** (pull from Artifact Registry or Docker Hub)

It’s written for Windows (PowerShell/CMD) + Ubuntu VMs, but notes for other setups are included.

---

## Part 1 — Connect VS Code to a GCP VM (Remote‑SSH)

### Prerequisites (local machine)

* VS Code installed
* VS Code extension: **Remote – SSH** (Microsoft)
* **Google Cloud SDK** installed and logged in: `gcloud auth login`
* **OpenSSH client** (included in recent Windows; otherwise enable “OpenSSH Client” in Optional Features)

### Prerequisites (GCP)

* A **Compute Engine VM** in your project
* Attach a Service Account that has received access to all this (tell @Beni to give access) (e.g., `Artifact Registry Reader`, `Storage Object Viewer/Admin`, `Secret Manager Secret Accessor`)

### A) Create the VM in the Console

1. Console → **Compute Engine → VM instances → Create instance**
2. Choose a region/zone.
3. **Machine type**: anything you like (e.g., `e2-medium`).
4. **Identity and API access** → pick your **Service Account** (the one with access to the bucket and registry); Access scopes: **Allow full access to all Cloud APIs**.
5. **Networking** → ensure you have an external IP (or plan to use IAP; see “No external IP” below).
6. Click **Create**.

### B) Get the VM external IP

In Windows PowerShell/CMD:

```bat
gcloud compute instances describe <VM_NAME> --zone <ZONE> --format="get(networkInterfaces[0].accessConfigs[0].natIP)"
```

In MAC/Linux terminal:

```bash
gcloud compute instances describe <VM_NAME> --zone <ZONE> --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```

Copy the IP.

### C) Prime SSH keys (JUST NEED TO DO IT ONCE)

Run once so `gcloud` generates your key and pushes the public key to the VM:

```bat
gcloud compute ssh <VM_NAME> --zone <ZONE>
```

Close that SSH session.

### D.1) Add an entry to your SSH config (Windows)

Create or edit: `C:\Users\<YOU>\.ssh\config`

```
Host gcp-dev
    HostName <EXTERNAL_IP>
    User <YOUR_LINUX_USERNAME>
    IdentityFile C:/Users/<YOU>/.ssh/google_compute_engine
    IdentitiesOnly yes
```

### D.2) (MAC/Linux) Add an entry to your SSH config
Create or edit: `~/.ssh/config`

```
Host gcp-dev
    HostName <EXTERNAL_IP>
    User <YOUR_LINUX_USERNAME>
    IdentityFile ~/.ssh/google_compute_engine
    IdentitiesOnly yes
```

* `<YOUR_LINUX_USERNAME>` = output of `whoami` inside the VM (you can run: `gcloud compute ssh <VM_NAME> --zone <ZONE> --command "whoami"`).

### E) Test SSH locally

```bat
ssh gcp-dev
```

If you get a shell on the VM, you’re good.

### F) Connect from VS Code

1. In VS Code press **F1** → **Remote‑SSH: Connect to Host…** → choose **gcp-dev**
2. A new VS Code window opens (remote). Use the Remote Explorer panel to open folders, terminals, etc.

### G) Install Docker on the VM 
SSH into the VM (VS Code Remote or terminal), then:

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER
newgrp docker
```

Verify:

```bash
docker run hello-world
```

### H) Run your container image
To run a container mounted locally for development, you can use:
```bash
export IMAGE_NAME="xxxx"
# Run Docker container with mounted volumes
docker run --rm --name $IMAGE_NAME -ti \
    -v "$(pwd)":/app \
    $IMAGE_NAME
```

### I) Push to Artifact Registry from the VM or from your local machine (optional)

```bash
# Authenticate Docker to use the Artifact Registry
gcloud auth configure-docker <REGION>-docker.pkg.dev

# Tag your image
docker tag $IMAGE_NAME us-east1-docker.pkg.dev/bitefinder-474614/bitefinder-images/$IMAGE_NAME

# Push your image
docker push us-east1-docker.pkg.dev/bitefinder-474614/bitefinder-images/$IMAGE_NAME
```

### END) Common fixes

* **Git not detected** in VS Code after installing:

  * F1 → **Remote‑SSH: Kill VS Code Server on Host** → reconnect
  * Or set **Remote Settings (SSH)** `"git.path": "/usr/bin/git"`
* **Quotes on Windows**: Use double quotes for `--format="..."` in gcloud commands.
* **Permission denied for Docker**:

  ```bash
  sudo usermod -aG docker $USER
  newgrp docker
  docker run hello-world
  ```
* **No external IP?** Use IAP tunneling. SSH config example:

  ```
  Host gcp-dev-iap
      HostName <VM_NAME>
      User <YOUR_LINUX_USERNAME>
      ProxyCommand /usr/bin/env bash -lc 'gcloud compute start-iap-tunnel %h 22 --project <PROJECT_ID> --zone <ZONE> --listen-on-stdin'
      IdentityFile C:/Users/<YOU>/.ssh/google_compute_engine
      IdentitiesOnly yes
  ```

  Then connect to `gcp-dev-iap` in VS Code.

---

## Part 2 — Spin up a VM to Run a Container Image

Two paths:

#### 1) Create the VM (Console)

* Compute Engine → **Create instance**
* Attach your **Service Account**
* Ensure external IP (or plan for IAP)
* Create

#### 2) Install Docker on the VM

SSH into the VM (VS Code Remote or terminal), then:

```bash
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker $USER
newgrp docker
```

Verify:

```bash
docker run hello-world
```

#### 3) Pull from **Artifact Registry** (recommended)

On the VM, configure Docker to use the Artifact Registry credential helper:

```bash
gcloud auth configure-docker <REGION>-docker.pkg.dev
```

> On a GCE VM, `gcloud` will use the VM’s **service account** behind the scenes. No secrets.json required.

Pull your image:

```bash
docker pull us-east1-docker.pkg.dev/bitefinder-474614/bitefinder-images/<IMAGE>:<TAG>
```

PROJECT_ID = bitefinder-474614
REPO = bitefinder-images

#### 4) Run the container

```bash
docker run --rm -it \
  -e GCP_BUCKET_NAME=bitefinder-data \
  -e GOOGLE_CLOUD_PROJECT=bitefinder-474614 \
  us-east1-docker.pkg.dev/bitefinder-474614/bitefinder-images/<IMAGE>:<TAG>
```

* If your code uses ADC (Google client libraries), it will authenticate via the VM’s **metadata server** using the attached service account.
* Make sure the service account has needed IAM: `roles/artifactregistry.reader` (to pull), and Storage roles on your bucket.


## IAM Quick Reference (what roles are needed)

* **Pull image from Artifact Registry**: `roles/artifactregistry.reader` on the **repository** (for the VM/service account and for coworkers)
* **Push image to Artifact Registry**: `roles/artifactregistry.writer`
* **Read from bucket**: `roles/storage.objectViewer`
* **Upload to bucket**: `roles/storage.objectCreator` (upload‑only) or `roles/storage.objectAdmin` (read/write objects)
* **Access Secret Manager**: `roles/secretmanager.secretAccessor` (if fetching secrets at runtime)

> Grant roles to **service accounts** (preferred) or user emails. Cross‑project: grant roles on your resource to their **service account** email.

---

## Appendices

### A) VS Code Remote‑SSH Troubleshooting

* Kill & restart the remote server: F1 → *Remote‑SSH: Kill VS Code Server on Host*
* Force Git path in Remote Settings (SSH): `"git.path": "/usr/bin/git"`
* Path issues? `which git`, `echo $PATH`
* Permissions: for Docker, add your user to `docker` group (above)

### B) Python Virtual Environment on the VM (avoid root pip warnings)

```bash
sudo apt-get install -y python3-venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### C) Read JSON from GCS instead of local files

```python
from google.cloud import storage
import json, os

bucket_name = os.getenv("GCP_BUCKET_NAME")
client = storage.Client()
bucket = client.bucket(bucket_name)

def read_json(path):
    data = bucket.blob(path).download_as_bytes()
    return json.loads(data)

locations = read_json("synthetic-data/locations.json")
symptoms = read_json("synthetic-data/symptoms.json")
```

### D) Fetch a secret from Secret Manager (no API keys in images)

```python
from google.cloud import secretmanager

def get_secret(project_id, secret_id):
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    resp = client.access_secret_version(name=name)
    return resp.payload.data.decode()
```

---

## Quick Checklist

* [ ] VM created, Service Account attached, external IP (or IAP)
* [ ] SSH config entry in `~/.ssh/config` (Windows path to `google_compute_engine` key)
* [ ] VS Code Remote‑SSH connects
* [ ] Docker installed (or COS image chosen)
* [ ] `gcloud auth configure-docker <REGION>-docker.pkg.dev`
* [ ] IAM: registry Reader/Writer, Storage roles, Secret Manager access if needed
* [ ] Container runs and can reach GCS via ADC (no secrets.json)
