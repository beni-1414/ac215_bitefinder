from PIL import Image
import argparse
import wandb
import os
import torch
from package.training.model import model_classes

def main():
    # Define inference arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('image_fp', type=str)
    parser.add_argument('text', type=str)

    # Parse inference arguments
    args = parser.parse_args()
    text = args.text
    image = Image.open(args.image_fp).convert('RGB')
    # 'antsimage391.jpg'
    # 'My arm was burning a few days ago and has been itching since.'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    api = wandb.Api()
    
    # Load W&B entity and project from env vars
    wandb_team = os.environ['WANDB_TEAM']
    wandb_project = os.environ['WANDB_PROJECT']

    # Download artifact
    artifact_root = wandb_team+'/'+wandb_project+'/'
    artifact_name = artifact_root+'labels_v2_20251101_182957:v0'
    artifact = api.artifact(artifact_name)
    artifact_dir = artifact.download()

    # Pull artifact metadata
    metadata = dict(artifact.metadata)
    model_id = metadata["model_id"]
    num_labels = metadata["num_labels"]
    id_to_label = metadata["id_to_label"]
    id_to_label = {int(k): v for k, v in id_to_label.items()} # Reconvert keys to ints

    # Instansiate model from saved artifact
    model_class = model_classes[model_id]
    model = model_class(num_labels=num_labels)
    model.model = model.model.from_pretrained(artifact_dir)
    model.processor = model.processor.from_pretrained(artifact_dir)
    model.classifier.load_state_dict(torch.load(f'{artifact_dir}/classifier.pt', map_location=torch.device(device)))
    
    # Run inference
    model.eval()
    processed = model.processor(text=[text], images=[image], return_tensors='pt', padding=True).to(device)
    with torch.no_grad():
        outputs = model(**processed)
        probs = torch.softmax(outputs['logits'], dim=-1)[0]
        pred = probs.argmax().item()
        print(f'Predicted: {id_to_label[pred]} ({probs[pred]:.2f})')
        print(f'Logits: {probs.tolist()}')

if __name__ == '__main__':
    main()