from typing import Optional
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, model_validator
from PIL import Image
from io import BytesIO
import base64
import wandb
import os
import torch
from google.cloud import storage
from api.package.training.model import model_classes
from api.package.training.utils_gcp import get_secret

router = APIRouter()

# Global variables
model = None
id_to_label = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'


@router.on_event('startup')
def load_model():
    global model, id_to_label, device

    # Set W&B API key from GCP Secret Manager
    wandb_key = get_secret('WANDB_API_KEY')
    if wandb_key:
        os.environ['WANDB_API_KEY'] = wandb_key

    api = wandb.Api()

    wandb_team = os.environ['WANDB_TEAM']
    wandb_project = os.environ['WANDB_PROJECT']
    artifact_root = wandb_team + '/' + wandb_project + '/'
    artifact_model_label = 'run_20251121_132516'  # TODO: make dynamic
    artifact_model_version = 'latest'
    artifact_name = artifact_root + artifact_model_label + ':' + artifact_model_version

    # Retrieve cache directory for model weights (or make if first time)
    cache_dir = os.getenv('MODEL_CACHE_DIR', '/tmp/vlmodel_cache')
    os.makedirs(cache_dir, exist_ok=True)

    # Download artifact from W&B into persistent cache if not already cached
    artifact_dir = os.path.join(cache_dir, artifact_name.replace('/', '_').replace(':', '_'))
    if not os.path.exists(artifact_dir):
        artifact = api.artifact(artifact_name)
        artifact.download(root=artifact_dir)

    # Pull artifact metadata
    artifact = api.artifact(artifact_name)
    metadata = dict(artifact.metadata)
    model_id = metadata['model_id']
    num_labels = metadata['num_labels']
    id_to_label = {int(k): v for k, v in metadata['id_to_label'].items()}

    # Instansiate model from saved artifact
    model_class = model_classes[model_id]
    model = model_class(num_labels=num_labels)
    model.model = model.model.from_pretrained(artifact_dir)
    model.processor = model.processor.from_pretrained(artifact_dir)
    model.classifier.load_state_dict(torch.load(f'{artifact_dir}/classifier.pt', map_location=torch.device(device)))

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
