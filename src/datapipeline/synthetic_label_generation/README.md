# Synthetic Label Generation

## Symptoms
Extracts subjective sensory symptoms for insect and arachnid bites from Cleveland Clinic website. Symptoms are categorized by frequency (common vs. rare/allergic reactions) and structured in JSON format with separate lists for each bite type.

## Locations
Generates plausible human-reported bite locations for 7 arthropod types. Creates 30 unique descriptions per type, mixing indoor/outdoor environments based on ecological knowledge and focusing on what people would reasonably remember when reporting bite incidents.

## label_generation.py
For every location, it generates 4 paraphrased versions including symptoms, vocab variability, shitty answers...

## Versions
### v1.0
Only contains info regarding symptoms and location. No mention of body part nor number of bites (one, many...) which would both be common in real world scenario. We need to decide if it is worth to feed the 1900 images through openAI api (about 2$/1000 images) and have it identify bodypart (or yield UNK which would allow us to randomly assign one) and the number of bites (one, many).

## Docker Setup

This directory contains a containerized version of the synthetic bite label generation system using `uv` for fast package management.

### Prerequisites

- Docker installed on your system
- OpenAI API key configured in the `.env` file at the repository root

### Files Overview

- `Dockerfile` - Docker image configuration using uv
- `docker-shell.sh` - Shell script for Linux/Mac to build and run the container  
- `docker-shell.bat` - Batch script for Windows to build and run the container
- `pyproject.toml` - Python project configuration and dependencies
- `uv.lock` - Locked dependency versions for reproducible builds

### Usage

**Linux/Mac:**
```bash
./docker-shell.sh
```

**Windows:**
```cmd
docker-shell.bat
```

**Manual Docker Commands:**
```bash
# Build and run
docker build -t synthetic-label-generation .
docker run --rm -it -v "$(pwd):/app" --env-file ../../../.env synthetic-label-generation
```

### Environment Variables

Make sure your `.env` file (located at the repository root) contains:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### Development

**Inside the container:**
Once inside the container shell, you MUST use `uv run` to activate the virtual environment:
```bash
# ✅ Correct - uses uv's virtual environment
uv run label_generation.py

# ❌ Wrong - will give "ModuleNotFoundError"
python label_generation.py
```

**Alternative inside container:**
```bash
# Manually activate the virtual environment
source .venv/bin/activate
python label_generation.py
```

**Local development:**
```bash
uv sync
uv run label_generation.py
```