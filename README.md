# bitefinder

Group Name: Bite Finder

Team Members: Jack Hwang, Zoe Zabetian, Irith Katiyar, Benet Fité

## Overview
As it is right now, the project contains 4 containerized workflows that achieve 4 different tasks. At the moment, they all use a GCP bucket as a shared storage space. Below is the specific documentation for each container

## Prerequisites
- Docker installed on your system
- OpenAI API key on your GCP account secrets manager called `OPENAI_API_KEY`, accessible to the service account you are using (only for `synthetic_label_generation`)
- A service account with access to the GCP bucket ``bitefinder_data``.

### Usage
To set up and run the containers, follow these steps:
#### Local
Running it from Windows (Git Bash) gives issues with mounting volumes. Use WSL or a Linux/Mac system. Also, ensure you have a secrets folder outside of the repo with a file called bitefinder-service-account.json containing the service account key.
```bash
sh src/{container_name}/docker-shell-local.sh
```
#### VM
Ensure the VM is authenticated with the proper service account.
```bash
sh src/{container_name}/docker-shell-vm.sh
```

## Container Workflows

### synthetic_label_generation
This container handles the generation of synthetic labels for training data. It uses predefined symptoms from literature and plausible bite locations (generated with GPT-5) to create diverse and realistic user reports. The output is a JSON file with a series of labels for each bite type.

It is a preprocessing step only to be run when the symptoms, locations or label generation logic changes. More detail in [src/synthetic_label_generation/README.md](src/synthetic_label_generation/README.md)

Once inside the container, run:
```bash
python3 src/synthetic_label_generation/label_generation.py
```

You will find the output both in the container and in the GCP bucket ``data/text/training/texts_{timestamp}.json``