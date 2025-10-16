import torch
from PIL import Image
from model import CLIPForBugBiteClassification, ViLTForBugBiteClassification, model_classes
from utils_save import *
from utils_gcp import *

config = load_config(save_dir)
model_class = config['model_class']
id_to_label = config['id_to_label']
num_labels = config['num_labels']

upload_to_gcp = True
pred_file = 'bite_prediction.txt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model_classes[model_class](num_labels=num_labels)
load_model(model, save_dir)

image = Image.open('data/bug-bite-images/testing/ants/fire_antsimage194.jpg').convert('RGB')
text = 'My arm is burning, I think I got bit by something at the beach.'

processed = model.processor(text=[text], images=[image], return_tensors='pt', padding=True).to(device)
with torch.no_grad():
    outputs = model(**processed)
    probs = torch.softmax(outputs['logits'], dim=-1)[0]
    pred = probs.argmax().item()
    print(f'Predicted: {id_to_label[pred]} ({probs[pred]:.2f})')
    if upload_to_gcp:
        with open(pred_file, 'w') as file:
            file.write(f'{id_to_label[pred].replace('_', ' ')}')
        upload_file_to_bucket(pred_file, pred_file)