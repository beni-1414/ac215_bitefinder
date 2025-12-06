from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, model_validator
from PIL import Image
from pillow_heif import register_heif_opener
from io import BytesIO
import base64
import wandb
import os
import shutil
import torch
from google.cloud import storage
from api.package.training.model import model_classes
from api.package.training.utils_secret import get_secret

register_heif_opener()  # Enables HEIC decoding

router = APIRouter()

# Global variables
model = None
id_to_label = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_loaded = False


REQUIRED_FILES = [
    "artifact.txt",
    "classifier.pt",
    "config.json",
    "merges.txt",
    "model.safetensors",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
]


def weights_cached(cache_dir: str) -> bool:
    for f in REQUIRED_FILES:
        if not os.path.isfile(os.path.join(cache_dir, f)):
            return False
    return True


def clear_cache(cache_dir: str):
    print("LOAD:     Clearing corrupted cache...")
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)
    print("DONE:     Created fresh cache.")


@router.on_event('startup')
def load_model():
    global model, id_to_label, device, model_loaded

    if model_loaded:
        return
    model_loaded = True

    # Retrieve W&B API key from secret manager
    print("LOAD:     Logging into W&B...")
    wandb_key = get_secret('WANDB_API_KEY')
    if wandb_key:
        os.environ['WANDB_API_KEY'] = wandb_key

    # Initialize W&B API
    api = wandb.Api()
    print("DONE:     Logged into W&B.")

    artifact_root = os.environ['WANDB_TEAM'] + '/' + os.environ['WANDB_PROJECT'] + '/'
    artifact_model_label = 'clip_20251128_225047'  # TODO: make dynamic
    artifact_model_version = 'v0'
    artifact_name = artifact_root + artifact_model_label + ':' + artifact_model_version

    # Retrieve cache directory for model weights (or make if first time)
    cache_dir = os.getenv('MODEL_CACHE_DIR', '/app/vlmodel_cache')
    os.makedirs(cache_dir, exist_ok=True)

    # Download artifact from W&B into persistent cache if not already cached
    if not weights_cached(cache_dir):
        clear_cache(cache_dir)
        print("LOAD:     Downloading model weights from W&B...")
        artifact = api.artifact(artifact_name)
        artifact.download(root=cache_dir)
        print("DONE:     Model weights downloaded.")

    # Pull artifact metadata
    artifact = api.artifact(artifact_name)
    metadata = dict(artifact.metadata)
    model_id = metadata['model_id']
    model_kwargs = metadata['model_kwargs']
    id_to_label = {int(k): v for k, v in metadata['id_to_label'].items()}

    # Instansiate model from saved metadata
    print("LOAD:     Instantiating model...")
    model_class = model_classes[model_id]
    model = model_class(**model_kwargs)
    print("DONE:     Model instantiated.")

    # Load model weights from saved artifact
    print(f"LOAD:     Loading weights into {model_id} model...")
    model.model = model.model.from_pretrained(cache_dir)
    print(f"LOAD:     Loading weights into {model_id} processor...")
    model.processor = model.processor.from_pretrained(cache_dir)
    print("LOAD:     Loading weights into classification head...")
    model.classifier.load_state_dict(torch.load(f'{cache_dir}/classifier.pt', map_location=torch.device(device)))
    print("DONE:     Model is served.")

    # Set model to eval mode
    model.eval()


'''
InferenceRequest: model for inference request payload (text + image)
'''


class InferenceRequest(BaseModel):
    # Text input
    text_raw: Optional[str] = None  # Raw text as string
    text_gcs: Optional[str] = None  # Path in bucket of text file (e.g., 'user_input/input.txt')
    # Image input
    image_base64: Optional[str] = None  # Raw image as base64-encoded string
    image_gcs: Optional[str] = None  # Path in bucket of image file (e.g., 'user_input/input.jpg')

    # Ensure exactly one source per input modality
    @model_validator(mode='after')
    def check_input(self):
        if (self.text_raw is None) == (self.text_gcs is None):  # Check if both text inputs are empty or given
            raise ValueError('you must provide only one text input: text_raw OR text_gcs.')
        if (self.image_base64 is None) == (self.image_gcs is None):  # Check if both image inputs are empty or given
            raise ValueError('you must provide only one image input: image_base64 OR image_gcs.')
        return self


@router.post('/predict')
def predict(request: InferenceRequest):
    global model, id_to_label, device
    image, text = None, None

    # Load text input
    if request.text_raw:
        # Raw text
        text = request.text_raw
    else:
        # Load text from GCS bucket
        storage_client = storage.Client()
        bucket = storage_client.bucket(os.getenv('GCP_BUCKET_NAME'))
        text_blob = bucket.blob(request.text_gcs)
        if not text_blob.exists():
            raise HTTPException(status_code=404, detail=f'Input error: {request.text_gcs} not found in bucket!')
        text = text_blob.download_as_text()

    # Load image input
    if request.image_base64:
        # Decode base64 image
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
    else:
        # Load image from GCS bucket
        storage_client = storage.Client()
        bucket = storage_client.bucket(os.getenv('GCP_BUCKET_NAME'))
        image_blob = bucket.blob(request.image_gcs)
        if not image_blob.exists():
            raise HTTPException(status_code=404, detail=f'Input error: {request.image_gcs} not found in bucket!')
        image_bytes = image_blob.download_as_bytes()
        image = Image.open(BytesIO(image_bytes)).convert('RGB')

    # Process inputs
    processed = model.processor(text=[text], images=[image], return_tensors='pt', padding=True).to(device)

    # Run inference and get predicted label and probabilities
    with torch.no_grad():
        outputs = model(**processed)
        probs = torch.softmax(outputs['logits'], dim=-1)[0]
        pred = probs.argmax().item()
        pred_label = id_to_label[pred]
        pred_conf = probs[pred].item()

        # Return prediction as JSON response
        return JSONResponse(
            {
                'prediction': pred_label,
                'confidence': round(pred_conf, 4),
                'probabilities': {id_to_label[i]: float(p) for i, p in enumerate(probs)},
            }
        )
