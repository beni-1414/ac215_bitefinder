# bitefinder

Group Name: Bite Finder

Team Members: Jack Hwang, Zoe Zabetian, Irith Katiyar, Benet Fité

## Overview
Our project is a bug bite classification app, where a user can take a photo of their skin on their phone and type their symptoms, and the app will predict what bug caused the bite and provide recommendations on which products to use or whether to see a doctor.

Currently, our project contains 4 containerized workflows that achieve 4 different tasks. At the moment, they all use a GCP bucket as a shared storage space.

## Prerequisites
- Docker installed on your system
- OpenAI API key on your GCP account secrets manager called `OPENAI_API_KEY`, accessible to the service account you are using (only for `synthetic_label_generation`)
- A service account with access to the GCP bucket ``bitefinder_data``.

## Data versioning
Since our data does not change ofter, we are not using a data versioning tool like DVC. Instead, we are storing all our data in a GCP bucket, and saving snapshots of the data whenever it changes with the pattern `data_v{version_number}`. That way, when we train a model we can specify exactly which version of the data we used and it is reproducible.

## Usage
To set up and run the containers, follow these steps:
### Local
Running it from Windows (Git Bash) gives issues with mounting volumes. Use WSL or a Linux/Mac system. Also, ensure you have a secrets folder outside of the repo with a file called `bitefinder-service-account.json` containing the service account key.
```bash
sh src/{container_name}/docker-shell.sh
```
or
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

It is a preprocessing step only to be run when the symptoms, locations or label generation logic changes.

Once inside the container, run:
```bash
python3 label_generation.py
```

You will find the output both in the container and in the GCP bucket ``data/text/training/texts_{timestamp}.json``

More detail about this container and its usage can be found in [src/synthetic_label_generation/README.md](src/synthetic_label_generation/README.md).

### image_augmentation
This container handles the augmentation of the raw images for preprocessing before training. It downloads the raw dataset from the GCP bucket, applies augmentations such as rotations and jitter with customizable parameters, and uploads the preprocessed images back up to the GCP bucket.

It is a preprocessing step only to be run once in this lifecycle.

Once inside the container, run:
```bash
python augment_images.py <args>
```

More detail about this container and its usage can be found in [src/image_augmentation/README.md](src/image_augmentation/README.md).

### vlmodel
This container is responsible for the vision-language classification model training and inference. For training, it downloads the image and text data from the GCP bucket, constructs a paired image-text dataset, runs model training/validation, and saves the model state. For inference, it loads the model from its saved state, passes a specified image and text input through the model, and outputs the predicted bug bite.

Once inside the container, to train a model:
```bash
python train.py
```

Once inside the container, to run inference:
```bash
python infer.py <image_fp> <text>
```

More detail about this container and its usage can be found in [src/vlmodel/README.md](src/vlmodel/README.md).

### ragmodel
This container implements a RAG system used to generate treatment advice and recommendations for the patient. The current version of this system uses `ChromaDB` with Cleveland Clinic webpages on each bug bite as the knowledge base. The system takes in as input the user's text query and the classification model's prediction, and outputs a natural-language recommendation for the user.

Once inside the container, run:
```bash
python cli.py --chat --chunk_type char-split \
  --symptoms <text> \
  --class <prediction>
```

More detail about this container and its usage can be found in [src/ragmodel/README.md](src/ragmodel/README.md).
