# Image Augmentation

This module performs **offline image augmentation** to expand the Kaggle bug-bite image dataset before text generation and model training.

## Directory Layout
```
image_augmentation/
├── augment_images.py # main augmentation script
├── generate_full_dataset.py # helper script that runs the full augmentation command
├── pyproject.toml # python dependencies and project metadata
├── Dockerfile # docker build file for reproducible environment
├── docker-shell.sh # helper script (mac/linux) to open docker container shell
├── docker-shell.bat # helper script (windows) to open docker container shell
├── output/ # full generated dataset (ignored by git)
└── output-samples/ # tiny demo dataset (safe to commit)
```

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


## Run Augmentation

### Option 1 — Run the Python helper
```bash
python generate_full_dataset.py
```

### Option 2 — Run the command directly from the shell
```bash
python augment_images.py \
  --input-dir ../../../../kaggle_bite_images/training \
  --output-dir ./output/training \
  --copies 2 \
  --image-size 224 \
  --strength medium \
  --include-original \
  --seed 42
```
### Adjustable Parameters

- `--input-dir`: path to original training data
- `--output-dir`: directory where augmented images will be written
- `--copies`: number of augmented versions per image
- `--image-size`: output size for random resized crop
- `--strength`:	augmentation intensity (light, medium, or strong)
- `--include-original`: if set, keeps a copy of each original image
- `--seed`: global seed for reproducible randomness


### Notes
Modify any argument inside `generate_full_dataset.py` to change defaults.

The augmented dataset will appear under: `image_augmentation/output/training/`