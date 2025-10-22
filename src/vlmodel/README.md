# Vision-Language Classification Model

The `vlmodel` container is responsible for the classification step in our pipeline, i.e., given an image and text, classify what bug bite the patient has. This container implements a paired image-text dataset from the processed images and synthesized text stored in our GCP bucket, two vision-language classification model architectures built on top of CLIP and ViLT, model training and validation, model saving/loading, and inference given a new image and text.

## Usage

To run the `vlmodel` container, navigate to the correct directory and run `docker-shell.sh`:
```bash
cd src/vlmodel
sh docker-shell.sh
```
This will build the image with the required dependencies and run the container, opening in `app`.

To train a model, run:
```bash
python train.py
```
Note: currently, the training script is not a CLI, so training arguments must be specified at the top of the script, such as the model type (`'clip'` or `'vilt'`), number of epochs, batch size, and learning rate.
The training script does the following:
1. Downloads the dataset from the GCP bucket.
2. Splits the dataset into training and validation sets.
3. Trains the model on the training set.
4. Evaluates the model the validation set.
5. Saves the model config and weights.
The training loss and accuracy per epoch as well as the evaluation loss and accuracy are printed.

To run inference
```bash
python infer.py <image_fp> <text>
```
The inference script does the following:
1. Load the model from its saved configuration and weights.
2. Process the image and text command line inputs.
3. Pass the inputs through the model and print the predicted bug bite.
4. Optionally upload the prediction to the GCP bucket.

## Container Structure

The primary class files in `vlmodel` are:
- `dataset`: labeled paired image-text dataset
- `model`: vision-language classification model architectures
- `trainer`: training and validation pipelines

The two runnable scripts are:
- `train`: train and validate model on training dataset
- `infer`: predict on given image and text

Finally, there are a series of `utils_` files with various helper functions, e.g., GCP bucket, model saving/loading, etc.

## Vision-Language Models

Currently, two vision-language model architectures are implemented: CLIP and ViLT. Both implement a classification head (a simple linear layer projecting the output embedding space to the label space) on top of the respective vision-language model, which has a majority of its parameters frozen. Both also implement a dropout layer to prevent overfitting.

## Data Format

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
