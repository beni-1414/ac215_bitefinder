import torch
from PIL import Image
from utils_save import *
from utils_gcp import *
from src.vlmodel.package.trainer.model import *
import argparse

gcp = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = load_config(dir=save_dir, from_gcp=gcp)
model_id = config['model_id']
id_to_label = config['id_to_label']
num_labels = config['num_labels']
model_class = model_classes[model_id]

parser = argparse.ArgumentParser()
parser.add_argument('image_fp', type=str)
parser.add_argument('text', type=str)

args = parser.parse_args()
text = args.text
image = Image.open(args.image_fp).convert('RGB')
# 'data/images/testing/ants/fire_antsimage194.jpg'
# 'My arm is burning, I think I got bit by something at the beach.'

model = model_class(num_labels=num_labels)
load_model(model, dir=save_dir, from_gcp=gcp)

processed = model.processor(text=[text], images=[image], return_tensors='pt', padding=True).to(device)
with torch.no_grad():
    outputs = model(**processed)
    probs = torch.softmax(outputs['logits'], dim=-1)[0]
    pred = probs.argmax().item()
    print(f'Predicted: {id_to_label[pred]} ({probs[pred]:.2f})')
    if gcp:
        pred_file = 'pred.txt'
        with open(pred_file, 'w') as file:
            file.write(f'{id_to_label[pred].replace('_', ' ')}')
        upload_file_to_bucket(pred_file, pred_file)