# Vision-Language Classification Model

The `vlmodel` container is responsible for the classification step in our pipeline, i.e., given an image and text, classify what bug bite the patient has. This container implements a paired image-text dataset from the processed images and synthesized text stored in our GCP bucket, two vision-language classification model architectures built on top of CLIP and ViLT, model training and validation, model saving/loading, and inference given a new image and text.

The `vlmodel` container is integrated with Weights & Biases for both experiment tracking and model weights storage. The Weights & Biases project is specified in the environment variables `WANDB_TEAM` and `WANDB_PROJECT`, and the `WANDB_API_KEY` with access to this project must be supplied in the GCP secret manager.

## Usage

### Inference

Simply run the `docker-shell.sh` script file to build the Docker image, run the container, and start the FastAPI app for inference:
```bash
sh docker-shell.sh
```

The API runs on port 9000, and its FastAPI docs can be viewed at http://0.0.0.0:9000/docs (once the server is running).

On startup, the service will download the metadata and model weights from the specified Weights & Biases artifact (currently hardcoded in `load_model`) and load the saved states into the appropriate model and processor. The model weights are downloaded to a persistent cache so future downloads of the same artifact can be bypassed. Instansiating the model architecture and loading its weights in make take some time depending on the model size and whether its a cold start.

The inference endpoint is `vlmodel/predict/` and it takes text and image inputs via the following string arguments in its request body:
- `text_raw` or `text_gcs`
- `image_base64` or `image_gcs`

The inference endpoint does the following:
1. Loads the text and image input (either raw or from the GCS bucket).
2. Processes the text and image.
3. Runs the model on the text and image.
4. Returns a prediction.

Predictions are returned in the following JSON format:
- `prediction`: predicted bug bite label
- `confidence`: probability (logits) for predicted label
- `probabilities`: probabilities (logits) for all labels

### Training

To run a training job serverlessly with Vertex AI:
```bash
sh package_training.sh # If any changes were made to the training package
sh run_training_job.sh
```

Training jobs can be run as a single training run or a hyperparameter sweep for further fine-tuning.

#### Single Training Run

Training jobs can be customized by editing the `job_config.yaml` file (located in the root `api/` folder).

The training job does the following:
1. Downloads the data from the GCP bucket.
2. Constructs a labeled paired image-text dataset.
3. Splits the dataset into training and validation sets.
4. Trains the model on the training set.
5. Evaluates the model the validation set.
6. Saves the model config and weights.

Training and validation metrics (loss, accuracy, confidence, etc.) are stored in a Weights & Biases run, and the associated model weights and metadata are stored in a Weights & Biases artifact named the same as the run. Currently, runs are named with the model name followed by a unique identifier (current datetime).

#### Hyperparameter Sweeps

Training jobs can also be configured to run hyperparameter sweeps. Parameters that you want to remain constant throughout the sweep can be specified in `job_config.yaml` as before, and parameters that you want to sweep over can be specified in `sweep.yaml` located in the training package (`api/package/training/sweep.yaml`). Note that since the sweep file is packaged with the training code, any changes to the sweep necessitate re-running `package-trainer.sh`. The sweep method (e.g., `grid`, `random`) is specified at the top of this file. Like a single training job, a hyperparameter sweep still runs as one Vertex AI custom job, i.e., the hyperparameter sweep runs sequentially on one provisioned machine. Thus, it is important to keep in mind the compute limits of the provisioned machine when working with large hyperparameter sweeps (e.g., run two separate "half" sweeps instead of one "full" sweep). Finally, it is not recommended to save (`--save`) artifacts when running a hyperparameter sweep as the free tier of Weights & Biases has very limited artifact storage and the artifact files for each vision-language model are quite hefty.

## Code Structure

The code in `vlmodel` is used for both model training and inference separately:

Model training is run serverlessly using Vertex AI. The training code is stored in `package/training`, and `job_config.yaml` specifies the parameters for the training job. The `setup.py` file in `package` lists the required dependencies. The scripts `package_training.sh` and `run_training_job.sh` are used to run a training job.

In the `training` package, `task.py` is the script that is run in the Vertex AI custom job. The primary classes it references are:
- `dataset`: labeled paired image-text dataset
- `model`: vision-language classification model architectures
- `trainer`: training and validation pipelines
- `utils_`: organized utility files with various helper functions

Model inference is run as an API service using FastAPI in a Docker container specified by `Dockerfile` and `pyproject.toml`/`uv.lock`. The Docker container is run with `docker-shell.sh`, which currently only supports development (interactive) mode: bash is opened at `app/` and secrets file is mounted at `secrets/`. Once in the container, the API can be run using `uvicorn` on the port specified in `docker-shell.sh`.

## Appendix

### Vision-Language Models

Currently, two vision-language model architectures are implemented: CLIP and ViLT. Both implement a customizable classification head (a set of linear layers with dropout and ReLU activation projecting the output embedding space to the label space) on top of the respective vision-language model, which can have a customizable number of layers frozen.

### Training Job Arguments

A training job can be customized with the following arguments:
- `model`: type of vl model, either "clip" or "vilt"
- `unfreeze_layers`: number of layers to unfreeze from pretrained vl model
- `classifier_layer`: number of linear layers in the classifier head on top of the vl model
- `activation`: type of activation function, either "relu" or "gelu"
- `dropout_prob`: dropout probability for dropout in classifier head
- `epochs`: number of training epochs (iterations)
- `batch_size`: batch size
- `optimizer`: optimizer class, either "sgd", "adam", or "adamw"
- `lr`: learning rate (for optimizer)
- `weight_decay`: weight decay (for optimizer)
- `text_fname`: the filename of the text inputs for the training data
- `data_root_dir`: the directory where the training data is housed in GCS bucket
- `seed`: seed to set for reproducibility
- `sweep_config`: config file provided only if hyperparameter sweep instead of a single training job
- `verbose`: flag set for extra printing
- `save`: flag set to save trained model weights to W&B
- `deploy`: flag set to deploy trained model into production

### Continuous Model Deployment

To deploy a trained model into production, both the `--save` and `--deploy` flags must be provided in the training job config, and the model must achieve at least an 80% accuracy on the validation set, to ensure consistently high model performance in our deployed app. The deploy flag simply updates the `best_model.txt` file in our GCS bucket with the new artifact name that has been stored on W&B, and our CD workflow is scheduled to re-up our `vlmodel` deployment if the name in this text file changes.

### Data Format

The dataset (whether stored locally or on a GCP bucket) must be formatted in the following directory structure, where `bug_1`, ..., `bug_n` correspond to the bug bite labels:
```bash
data/
    image/
        training/
            bug_1/
                img_1.jpg
                img_2.jpg
                ...
            bug_2/
            ...
            bug_n/
        testing/
            ...
    text/
        training/
            texts.json
        testing/
            texts.json
```
