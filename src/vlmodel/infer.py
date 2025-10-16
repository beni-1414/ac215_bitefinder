import torch
from PIL import Image
from utils_save import *
from utils_gcp import *
from utils_model import model_classes

gcp = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = load_config(dir=save_dir, from_gcp=gcp)
model_class = config['model_class']
id_to_label = config['id_to_label']
num_labels = config['num_labels']

model = model_classes[model_class](num_labels=num_labels)
load_model(model, dir=save_dir, from_gcp=gcp)

image = Image.open('data/images/testing/ants/fire_antsimage194.jpg').convert('RGB')
text = 'My arm is burning, I think I got bit by something at the beach.'

processed = model.processor(text=[text], images=[image], return_tensors='pt', padding=True).to(device)
with torch.no_grad():
    outputs = model(**processed)
    probs = torch.softmax(outputs['logits'], dim=-1)[0]
    pred = probs.argmax().item()
    print(f'Predicted: {id_to_label[pred]} ({probs[pred]:.2f})')
    # if gcp:
    #     pred_file = 'bite_prediction.txt'
    #     with open(pred_file, 'w') as file:
    #         file.write(f'{id_to_label[pred].replace('_', ' ')}')
    #     upload_file_to_bucket(pred_file, pred_file)