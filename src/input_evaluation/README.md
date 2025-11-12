# eval-microservice

Stateless FastAPI microservice for evaluating:
- **Text completeness** using Vertex AI (Gemini 1.5 Flash/Pro or "small").
- **Image quality** heuristics for bug-bite photos (no diagnosis, no enhancement).

## Endpoints

### POST /v1/evaluate/text
Request (JSON):
```json
{
  "user_text": "string",
  "first_call": true,
  "history": ["..."],
  "return_combined_text": true,
  "debug": false
}
````

Response:

```json
{
  "complete": true,
  "improve_message": null,
  "combined_text": null,
  "high_danger": false,
  "latency_ms": 123
}
```

### POST /v1/evaluate/image

**JSON (preferred)**:

```json
{ "image_gcs_uri": "gs://bucket/path.jpg" }
```

Response:

```json
{
  "usable": false,
  "improve_message": "Hold steady and re-take in brighter light.",
  "metrics": {
    "blur_laplacian_var": 28.7,
    "exposure_hist_entropy": 2.1,
    "under_over_exposed_ratio": 0.34,
    "noise_estimate_sigma": 24.2,
    "compression_artifacts_score": 0.71,
    "motion_blur_index": 0.62,
    "skin_patch_detected": true,
    "skin_area_ratio": 0.18,
    "exif_orientation": 1
  },
  "latency_ms": 95,
  "source": "gcs|upload"
}
```

## Environment

This service reads env from your existing `env.dev` two folders up and secrets via a path declared in that file. Relevant vars:

* Server: `PORT`, `LOG_LEVEL`, `ALLOW_ORIGINS`
* Vertex: `GOOGLE_CLOUD_PROJECT`, `VERTEX_REGION`, `VERTEX_MODEL_NAME` (e.g. `gemini-2.5-flash`), `GOOGLE_APPLICATION_CREDENTIALS` (local only)
* Thresholds (tuned via dataset):

  * `MIN_LAPLACIAN_VAR=60`
  * `MIN_EXPOSURE_ENTROPY=3.0`
  * `MAX_EXPOSURE_CLIP_RATIO=0.25`
  * `MAX_NOISE_SIGMA=20`
  * `MAX_BLOCKINESS=0.60`
  * `MAX_MOTION_BLUR=0.55`
  * `MIN_SKIN_AREA_RATIO=0.15`

## Things to be checked in frontend:
- Max image size before uploading to GCP.

## Local dev

```bash
bash docker-shell.sh
```

## Threshold tuning workflow

1. Collect a folder (or GCS prefix) of labeled images (good/bad optional).
2. Run the CLI:

```bash
python -m app.services.image_quality --input path/to/images --out out.csv
```

3. Inspect percentiles; set env thresholds accordingly.

## Tests
Run unit tests with:
```bash
python3 -m pytest --cov=app --cov-report=term-missing
```

Ensure pytest and pytest-cov are installed on your environment.