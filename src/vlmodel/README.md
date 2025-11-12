# Vision-Language Classification Model

The `vlmodel` container is responsible for the classification step in our pipeline, i.e., given an image and text, classify what bug bite the patient has. This container implements a paired image-text dataset from the processed images and synthesized text stored in our GCP bucket, two vision-language classification model architectures built on top of CLIP and ViLT, model training and validation, model saving/loading, and inference given a new image and text.

## Usage

### Inference

To build the Docker image and run the container for the inference server:
```bash
sh docker-shell.sh
```

Once in the interactive container, to start the FastAPI app for inference, run:
```bash
uvicorn inference_service:app --host 0.0.0.0 --port 8080
```

On startup, the service will download the model weights from the specified Weights & Biases artifact (currently hardcoded in `load_model`) and load the saved states into the appropriate model and processor. (This may take some time depending on if it is a cold start.)

The API runs on port 8080, and its documentation is available at http://127.0.0.1:8080/docs.

The inference endpoint is `predict/` and it takes text and image inputs via the following string arguments in its request body:
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

Training jobs can be customized by editing the `job_config.yaml` file.

The training job does the following:
1. Downloads the data from the GCP bucket.
2. Constructs a labeled paired image-text dataset.
3. Splits the dataset into training and validation sets.
4. Trains the model on the training set.
5. Evaluates the model the validation set.
6. Saves the model config and weights.

Training and validation metrics (loss, accuracy, etc.) are stored in a Weights & Biases run, and the associated model weights and metadata are stored in a Weights & Biases artifact.

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

Currently, two vision-language model architectures are implemented: CLIP and ViLT. Both implement a classification head (a simple linear layer projecting the output embedding space to the label space) on top of the respective vision-language model, which has a majority of its parameters frozen. Both also implement a dropout layer to prevent overfitting.

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
