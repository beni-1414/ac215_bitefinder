from PIL import Image
import argparse
import wandb
import torch

def main():
    print(f"🚀 Starting inference job...")

    # Define inference arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('image_fp', type=str)
    parser.add_argument('text', type=str)

    # Parse inference arguments
    args = parser.parse_args()
    text = args.text
    image = Image.open(args.image_fp).convert('RGB')
    # 'data/images/testing/ants/fire_antsimage194.jpg'
    # 'My arm is burning, I think I got bit by something at the beach.'

    api = wandb.Api()
    
    # Download artifact
    artifact_name = ''
    artifact = api.artifact(artifact_name)
    artifact_dir = artifact.download()

    # Pull artifact metadata
    metadata = dict(artifact.metadata)
    model_id = metadata["model_id"]
    num_labels = metadata["num_labels"]
    id_to_label = metadata["id_to_label"]

    # Instansiate model from saved artifact
    model_class = model_classes[model_id]
    model = model_class(num_labels=num_labels)
    model.model = model.model.from_pretrained(artifact_dir)
    model.processor = model.processor.from_pretrained(artifact_dir)
    model.classifier.load_state_dict(torch.load(f'{artifact_dir}/classifier.pt'))
    
    # Run inference
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processed = model.processor(text=[text], images=[image], return_tensors='pt', padding=True).to(device)
    with torch.no_grad():
        outputs = model(**processed)
        probs = torch.softmax(outputs['logits'], dim=-1)[0]
        pred = probs.argmax().item()
        print(f'Predicted: {id_to_label[pred]} ({probs[pred]:.2f})')

if __name__ == '__main__':
    main()