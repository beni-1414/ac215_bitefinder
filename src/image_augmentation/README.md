# Image Augmentation

This container performs image augmentation for the BiteFinder project. It downloads the training dataset from Google Cloud Storage, applies augmentations to expand the dataset, and uploads the processed images back to the bucket.

## 1. Set up the Docker container

Before running, ensure your Google Cloud credentials JSON file is available one directory above the project directory. Then, from the root of the `ac215_bitefinder` repository, run:
```bash
bash docker-shell.sh
```
This script:
- Builds the Docker image defined in the Dockerfile
- Creates a container named image-augmentation
- Mounts your working directory and secrets folder
- Sets up the environment variables defined in env.dev
- Opens an interactive shell in the container

## 2. Run the image augmentation script

```bash
python augment_images.py \
  --input-dir <bucket data> \
  --output-dir ./output/training \
  --copies 2 \
  --image-size 224 \
  --strength medium \
  --include-original \
  --seed 42
```

This command applies multiple augmentation operations (e.g., rotation, blur, color jitter, cropping) to each image and saves the results in src/image_augmentation/output/ (will upload to GCP bucket)


### Adjustable Parameters

- `--input-dir`: path to original training data
- `--output-dir`: directory where augmented images will be written
- `--copies`: number of augmented versions per image
- `--image-size`: output size for random resized crop
- `--strength`:	augmentation intensity (light, medium, or strong)
- `--include-original`: if set, keeps a copy of each original image
- `--seed`: global seed for reproducible randomness


## Expected Input Format
```kaggle_bite_images/
  training/
    ants/
      img1.jpg
      img2.jpg
    fleas/
      img1.jpg
      ...
  testing/
    ants/
    fleas/
    ...
```

Each class (`ants`, `fleas`, etc) must be its own folder containing images. The augmentation script preserves this structure in its output.