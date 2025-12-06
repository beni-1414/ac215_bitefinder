IMPORTANT: PLACE THE TWO JSONS FROM WHATSAPP IN YOUR SECRETS FOLDER. IF you get SSH related errors, follow the instructions in the dlops-readme to setup your gcloud environment.

Run `gcloud container clusters get-credentials bitefinder-cluster --region us-central1 --project bitefinder-474614` each time you restart the deployment Docker shell to setup `kubectl`.

NOTE: The vlmodel image is currently set to a fixed version, as it takes at least 30 mins to upload to GAR. To upload a new vlmodel image, don't use `deploy_images` like for the other images, manually upload it to GAR using:
- `docker build --platform=linux/amd64 -t us-east1-docker.pkg.dev/bitefinder-474614/bitefinder-images/vlmodel:dev .`
- `docker push us-east1-docker.pkg.dev/bitefinder-474614/bitefinder-images/vlmodel:dev`

STEPS TO START DEBUGGING POD:
1. Run `kubectl get pods -n bitefinder-namespace` to get pod names and see which one(s) have a status other than "Running".
2. Run `kubectl describe pod <pod_name> -n bitefinder-namespace` on the pod that is not working.
3. Run `kubectl logs <pod_name> -n bitefinder-namespace` on the pod that is not working.

STEPS TO DELETE PVC:
1. Get the PVC name by running `kubectl get pvc -n bitefinder-namespace`.
2. Run `kubectl delete pvc <pvc_name> -n bitefinder-namespace`.
3. If this command is hanging, in another terminal check PVC status try running `kubectl get pvc vlmodel-cache-pvc -n bitefinder-namespace -o wide`. If the status is Terminating, it is likely has a finalizer blocking deletion, so run `kubectl patch pvc vlmodel-cache-pvc -n bitefinder-namespace -p '{"metadata":{"finalizers": []}}' --type=merge` to safely remove it. The `kubectl delete pvc...` command should immediately finish now.
4. While the PVC is now deleted, some of the pods may still be referencing the PVC name, so the workaround to this is to simply change the PVC name: change `"vlmodel-pvc"` in `setup_containers.py` and then rerun `pulumi up --stack dev`.
