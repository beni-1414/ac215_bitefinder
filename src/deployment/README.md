# Deployment Instructions

This container is used to deploy the BiteFinder app on Google Kubernetes Engine (GKE), wether manaually or through GitHub Actions.

## Prerequisites

1. Place the jsons for the `deployment` and `gcp-service` service accounts in the `secrets` folder. Set them up according to the instructions in the main README of the project.
2. Run `docker-shell.sh` and shell into the container.
3. run `gcloud compute project-info add-metadata --project <YOUR GCP_PROJECT> --metadata enable-oslogin=TRUE` to enable OS login for the GCP project.
4. Add the SSH keys:
```
cd /secrets
ssh-keygen -f ssh-key-deployment
cd /app
```
5. Run `gcloud compute os-login ssh-keys add --key-file=/secrets/ssh-key-deployment.pub`

## Deployment
Run `gcloud container clusters get-credentials <cluster-name> --region us-central1 --project bitefinder-474614` each time you restart the deployment Docker shell to setup `kubectl`. Cluster-name can be bitefinder-cluster or bitefinder-agent-cluster.

NOTE: The vlmodel image is currently set to a fixed version, as it takes at least 30 mins to upload to GAR. To upload a new vlmodel image (which rarely needs to be done), you don't use `deploy_images` like for the other images, but manually upload it to GAR using:
- `docker build --platform=linux/amd64 -t us-east1-docker.pkg.dev/bitefinder-474614/bitefinder-images/vlmodel:dev .`
- `docker push us-east1-docker.pkg.dev/bitefinder-474614/bitefinder-images/vlmodel:dev`


### Build and Push Docker Containers to GCR
- cd into `deploy_images`
- When setting up pulumi for the first time run:
```
pulumi stack init dev/agent
pulumi config set gcp:project <project-name> --stack dev
```

This will save all your deployment states to a GCP bucket

- If a stack has already been setup, you can preview deployment using:
```
pulumi preview --stack dev/agent --refresh
```

- To build & push images run (This will take a while since we need to build 3 containers):
```
pulumi up --stack dev/agent -y
```

## Create & Deploy Cluster
- cd into `deploy_k8s` from the `deployment` folder
- When setting up pulumi for the first time run:
```
pulumi stack init dev/agent
pulumi config set gcp:project <project-name>
pulumi config set security:gcp_service_account_email deployment@<project-name>.iam.gserviceaccount.com --stack dev/agent
pulumi config set security:gcp_ksa_service_account_email gcp-service@<project-name>.iam.gserviceaccount.com --stack dev/agent
```
This will save all your deployment states to a GCP bucket

- If a stack has already been setup, you can preview deployment using:
```
pulumi preview --stack dev --refresh
```

- To create a cluster and deploy all our container images run:
```
pulumi up --stack dev --refresh -y
```

### STEPS TO START DEBUGGING POD:
1. Run `kubectl get pods -n bitefinder-namespace` to get pod names and see which one(s) have a status other than "Running".
2. Run `kubectl describe pod <pod_name> -n bitefinder-namespace` on the pod that is not working.
3. Run `kubectl logs <pod_name> -n bitefinder-namespace` on the pod that is not working.

### STEPS TO DELETE PVC:
1. Get the PVC name by running `kubectl get pvc -n bitefinder-namespace`.
2. Run `kubectl delete pvc <pvc_name> -n bitefinder-namespace`.
3. If this command is hanging, in another terminal check PVC status try running `kubectl get pvc vlmodel-pvc -n bitefinder-namespace -o wide`. If the status is Terminating, it is likely has a finalizer blocking deletion, so run `kubectl patch pvc vlmodel-pvc -n bitefinder-namespace -p '{"metadata":{"finalizers": []}}' --type=merge` to safely remove it. The `kubectl delete pvc...` command should immediately finish now.
4. While the PVC is now deleted, some of the pods may still be referencing the PVC name, so the workaround to this is to simply change the PVC name: change `"vlmodel-pvc"` in `setup_containers.py` and then rerun `pulumi up --stack dev/agent`.
