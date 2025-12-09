# bitefinder

Group Name: Bite Finder

Team Members: Jack Hwang, Zoe Zabetian, Irith Katiyar, Benet Fité

## Overview

BiteFinder is an AI-powered web app that helps users identify bug bites and receive tailored treatment advice. By combining a user's image and text data through a multimodal ML pipeline, it classifies the likely bug type and retrieves relevant medical guidance using a RAG model.

## MS 4

All of the required documentation for MS 4 is contained in this README file (e.g., application design document, local setup and usage, data versioning, model fine-tuning summary), or linked from this README file (e.g., solution and technical architecture and screenshots of CI run in `docs` folder, container README files).

## Usage

### Prerequisites

- Docker installed on your system
- Service account has access to GCP secret manager with the secrets `OPENAI_API_KEY`, `WANDB_API_KEY`, and `PINECONE_API_KEY`
- Service account has access to GCS bucket `bitefinder_data`

### How to Run Locally

1. Navigate to the `src` directory.
2. Run `docker compose build` to build all of the images.
3. Run `docker compose up` to run all of the containers.
4. Type `http://localhost:3001/chat` in your browser!

(Note that vlmodel application startup takes some time to complete.)

A demo of our app is located in the [docs/frontend_demo](docs/frontend_demo) folder, which shows a series of screenshots navigating through our app's flow.

## Application Design

Our application design follows a microservices architecture, separated into the following containers:
- `input_evaluation`: serves input evaluator model that checks the quality of user's image and text inputs
- `vlmodel`: serves vision-language classification model for bug bite prediction
- `ragmodel`: serves RAG system used to generate treatment advice and recommendations for the user
- `orchestrator`: serves orchestrator that performs chatting Q/A functionality and calls all prior services
- `frontend`: implements the React-based frontend of our application and calls orchestrator service on backend

Each container is organized as a stateless FastAPI container with an `api/` folder and a `tests/` folder (except the frontend, which is organized as a standard Next.js app). Each container is Dockerized, with a `Dockerfile` and dependency file (`pyproject.toml` for everything except the frontend that has `package.json`). Each container also has a `docker-shell.sh` file for local development of each individual container.

The general flow of our application is that when a user sends a message in the chat, the frontend will call the orchestrator service, which is responsible for calling the input evaluation, prediction, and RAG services as appropriate. When the user sends the initial image and text, the orchestrator calls the input evaluation service, and if of good quality, calls the VL model service that returns a prediction. When the user is sending follow-up questions, the orchestrator's LLM engages with Q/A with the user and decides whether/when to call the RAG tool.

Our application's solution architecture and technical architecture are described in the presentation here: [docs/solution-architecture.pdf](docs/solution-architecture.pdf).

### input_evaluation

This container implements a stateless FastAPI web service that evaluates the input of the user's image and text data. For the image, it checks various metrics that assess its quality. For the text, it checks both the user's symptoms/location text (the text the user sends with the image), as well as the user's follow-up questions (texts the user sends after receiving the prediction). It checks the completeness of the message the user writes (i.e., does it include symptoms and the location), and it checks the relevance of the user's follow-up questions.

More detail about this container and its individual usage can be found in [src/input_evaluation/README.md](src/input_evaluation/README.md).

### vlmodel

This container implements a stateless FastAPI web service that serves the vision-language model for bug bite prediction. Upon startup, it downloads the model weights and metadata from W&B of the specified "best" artifact and caches them in a persistent volume (if not already cached), then instansiates the vision-language model instance and loads the saved weights into the model. The single endpoint returns the model's prediction given the user's image and text.

More detail about this container and its individual usage can be found in [src/vlmodel/README.md](src/vlmodel/README.md).

### ragmodel

This container implements a stateless FastAPI web service that hosts a RAG system using Pinecone to generate treatment advice and recommendations for the patient. The system takes in as input the user's text query and the classification model's prediction, and outputs a natural-language recommendation for the user.

More detail about this container and its individual usage can be found in [src/ragmodel/README.md](src/ragmodel/README.md).

### orchestrator

This container implements a FastAPI web service that is responsible for the chatting Q/A with the user. We currently have two equivalent implementations for our orchestrator: a fully hard-coded version that controls the logic of when each of the services is called, and an agentic version that controls the follow-up Q/A and when our RAG service is called. The orchestrator serves as the layer between the frontend and the above backend services.

More detail about this container and its individual usage can be found in [src/orchestrator/README.md](src/orchestrator/README.md).

### frontend

The frontend contains our React-based app using Next.js. It contains all of the necessary UI modules for our chat. We are currently in the process of refining our UI, but our current version has a single page with a chat box where the user can send text and an image and engage with our AI expert with questions.

The frontend source code in `src` is organized into four subdirectories: `app` contains the main chat page, app layout, and global style sheet; `components` contains React components for basic user interface features (buttons, text areas, etc.), chat-specific components (chat input, chat message, and question suggestions), and layout (header, footer, light/dark mode toggle); `hooks`; and `lib` contains our `DataService` which integrates our frontend with our backend by making calls to the orchestrator service.

## ML Pipeline

Our application also houses a workflow for data processing and model training, composed of the following containers:
- `synthetic_label_generation`: generates synthetic labels for training data
- `image_augmentation`: augments the raw training images
- `vlmodel`: packages and runs a Vertex AI training job to train the vision-language model

While these containers aren't currently chained together, they will be in the near future in a Vertex AI Pipeline where synthetic label generation and image augmentation are both chained with VL model training. Note that model deployment is not chained in this pipeline because our model is not deployed as a Vertex AI endpoint but rather on our self-hosted container (as described above). However, we will still have continuous deployment for our model as a hook will trigger the `vlmodel` inference service to re-up itself whenever the model weights (on W&B) are updated. This workflow will be implemented by our final milestone.

[Note that the `vlmodel` directory contains both the code for model training and model inference, as they both need to share the same model architectures defined in the training package. The Dockerfile in `vlmodel` defines the container for serving the model for inference, while the training code is packaged in the `package` folder to be run serverlessly using Vertex AI.]

### synthetic_label_generation

This container handles the generation of synthetic labels for training data. It uses predefined symptoms from literature and plausible bite locations (generated with GPT-5) to create diverse and realistic user reports. The output is a JSON file with a series of labels for each bite type. It is a preprocessing step only to be run when the symptoms, locations or label generation logic changes.

More detail about this container and its individual usage can be found in [src/synthetic_label_generation/README.md](src/synthetic_label_generation/README.md).

### image_augmentation

This container handles the augmentation of the raw images for preprocessing before training. It downloads the raw dataset from the GCP bucket, applies augmentations such as rotations and jitter with customizable parameters, and uploads the preprocessed images back up to the GCP bucket. It is a preprocessing step only to be run once in this lifecycle, or re-run if the augmentation logic changes.

More detail about this container and its individual usage can be found in [src/image_augmentation/README.md](src/image_augmentation/README.md).

### vlmodel

In addition to the inference container serving the trained model, this directory contains scripts and a training package (`package/training`) that package and run a custom Vertex AI training job to train our vision-language classification model. The job either runs a single training job or a hyperparameter sweep. It downloads the image and text data from our GCS bucket, constructs a paired image-text dataset, runs model training/validation for the given set(s) of hyperparameters, logs training/validation metrics on W&B, and saves the model weights and metadata on W&B.

More detail about this container and its individual usage can be found in [src/vlmodel/README.md](src/vlmodel/README.md).

## Continuous Integration

Our continuous integration pipeline is defined in `.github/workflows/ci.yml`. Whenever a developer pushes code to the main branch of our GitHub repository, our GitHub Actions workflow is triggered, which builds each service image, runs linting and formatting on them, runs its tests, and generates a coverage report. Our formatter is Black and our linter is Flake8.

Unfortunately (since we are on the free tier of GitHub), whenever we run our CI pipeline now we get an error on GitHub Actions stating: `Artifact storage quota has been hit. Unable to upload any new artifacts.` when building each image. We have tried clearing the artifact storage and waiting 24 hours but we still get this error and we don't know how to fix it. Our last successful run of our CI pipeline is [here](docs/ci_pipeline_sample_run.png).

## Testing

### Unit and Integration Tests

Each backend service container contains a `tests` folder which has `unit` and `integration` tests in their respective subdirectories.

To run tests locally:
1. Navigate to `src/<service-name>` (e.g., `src/vlmodel`).
2. Run `pytest --cov=api --cov-report=term-missing` to run the tests and generate a coverage report.

Screenshots of passing tests and coverage reports can be found in [docs/test_coverage](docs/test_coverage) folder for each backend service container. All of our test suites exceed the 75% coverage requirement.

### System Tests

Follow the instructions in [system_tests.md](src/system_tests.md) to run end-to-end tests on our application endpoints using the FastAPI docs interface.

## Model Fine-Tuning

We experimented with two vision-language model architectures: CLIP and ViLT. Both models had their final layer replaced with our classification head to classify the one of seven bug bite types in our dataset. This classification head had linear layers with ReLU activation and dropout between. To experiment with the model architecture beyond just CLIP versus ViLT, we experimented with the number of linear layers in our classification head as well as the number of layers of the base model that we unfroze.

We performed a grid search over the following set of hyperparameters:
- Base model: CLIP, ViLT
- Number of layers unfrozen in base: 1, 2, 3, 4
- Number of layers in classification head: 1, 2, 3

We kept the following training hyperparameters constant in the search space:
- Epochs: 10
- Batch size: 32
- Optimizer: Adam
- Loss function: cross entropy
- Learning rate: 1e-4
- Dropout: 0.1
- Data version: latest

We recorded metrics on both training and validation sets (80%-20% split), including training loss, validation loss, training accuracy, validation accuracy, and validation average correct confidence (which is the mean of the confidences given to the correct labels). A "good" model had both a high validation accuracy (picking the right answer most of the time) and a high validation average correct confidence (picking the right answer with high confidence). Some models had a high accuracy but low confidence, meaning they would always pick the right answer but by a very narrow margin. We wanted to avoid these models, so we did not necessarily pick the model with the highest validation accuracy.

The best performing model was CLIP with 1 layer unfrozen and 3 layers in its classification head. It reached an accuracy of 94.0% and average correct confidence of 94.8% on the validation set. (The best ViLT model had 4 layers unfrozen and a classification head with 1 layer. It reached an accuracy of 86.9% and average correct confidence of 93.0% on the validation set.)

Experiment results are located in the [docs/training_experiments](docs/training_experiments) folder, which contains select screenshots from Weights & Biases. Some observations and general trends to note:
- The better CLIP models generally seemed to perform better than the better ViLT models, but ViLT generally had a higher average standard of performance compared to CLIP out of the entire search space of hyperparameters
- In the CLIP models, as the number of unfrozen layers increased, the model performance generally increased.
- In the CLIP models, as the number of classifier layers increased, the model performance generally increased.

## Data Versioning

Since our data does not change ofter, we are not using a data versioning tool like DVC. Instead, we are storing all our data in a GCP bucket, and saving snapshots of the data whenever it changes with the pattern `data_v{version_number}`. That way, when we train a model we can specify exactly which version of the data we used and it is reproducible.

### Raw Data

The raw image data used for bug bite classification is available here: https://www.kaggle.com/datasets/moonfallidk/bug-bite-images. The directory structure of this dataset is what is followed/expected by our trainer.

## Continuous Deployment
Push a commit to the main branch with the text `/deploy-app` to trigger a redeployment of the entire application. This will rebuild and redeploy all of the containers with the latest code. Workflow code is in `.github/workflows/cd.yml`.
