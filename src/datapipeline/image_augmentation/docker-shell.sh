#!/usr/bin/env bash
docker run --rm -it \
  -v "$(pwd)":/app \
  -v "$(pwd)/../raw_data":/data_in:ro \
  -v "$(pwd)/output":/data_out \
  image-augmentation:latest \
  -c "source /.venv/bin/activate && bash"
