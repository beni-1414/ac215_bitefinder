from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import base64
import wandb
import os
import torch
from package.training.model import model_classes

class InferenceRequest(BaseModel):
    text: str
    image_base64: str

router = APIRouter()

# Global variables
model = None
id_to_label = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@router.on_event("startup")
def load_model():
    global model, id_to_label, device
    
    # Download artifact from W&B
    api = wandb.Api()
    wandb_team = os.environ['WANDB_TEAM']
    wandb_project = os.environ['WANDB_PROJECT']
    artifact_root = wandb_team+'/'+wandb_project+'/'
    artifact_name = artifact_root+'labels_v2_20251101_182957:v0' # TODO: make dynamic
    artifact = api.artifact(artifact_name)
    artifact_dir = artifact.download()

    # Pull artifact metadata
    metadata = dict(artifact.metadata)
    model_id = metadata["model_id"]
    num_labels = metadata["num_labels"]
    id_to_label = {int(k): v for k, v in metadata["id_to_label"].items()}

    # Instansiate model from saved artifact
    model_class = model_classes[model_id]
    model = model_class(num_labels=num_labels)
    model.model = model.model.from_pretrained(artifact_dir)
    model.processor = model.processor.from_pretrained(artifact_dir)
    model.classifier.load_state_dict(torch.load(f'{artifact_dir}/classifier.pt', map_location=torch.device(device)))
    model.eval()

@router.post("/predict")
def predict(request: InferenceRequest):
    global model, id_to_label, device

    text = request.text
    image_bytes = base64.b64decode(request.image_base64)
    image = Image.open(BytesIO(image_bytes)).convert('RGB')

    processed = model.processor(text=[text], images=[image], return_tensors='pt', padding=True).to(device)
    with torch.no_grad():
        outputs = model(**processed)
        probs = torch.softmax(outputs['logits'], dim=-1)[0]
        pred = probs.argmax().item()
        pred_label = id_to_label[pred]
        pred_conf = probs[pred].item()

        return JSONResponse({
            "prediction": pred_label,
            "confidence": round(pred_conf, 4),
            "probabilities": {id_to_label[i]: float(p) for i, p in enumerate(probs)}
        })