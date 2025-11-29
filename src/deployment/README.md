CURRENT ISSUES:
1. Frontend is failing to start for some unknown reason. need to check the logs with kubectl logs <pod-name> -c frontend
2. Maybe other issues that will arise.

4. Set up a PVC cache like they did for the chroma db so that the vlmodel does not need to redownload the model every time the container restarts. It is halfway done Irith you might like to battle with this one.

IMPORTANT: PLACE THE TWO JSONS FROM WHATSAPP IN YOUR SECRETS FOLDER. IF you get SSH related errors, follow the instructions in the dlops-readme to setup your gcloud environment.

NOTE: the vlmodel image is currently set to a fixed version, it took about 30 mins to upload to GCP.
