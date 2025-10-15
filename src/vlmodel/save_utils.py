import os
import torch

def save_model(model, dir):
    if not os.path.exists(dir): os.makedirs(dir)
    model.model.save_pretrained(dir)
    model.processor.save_pretrained(dir)
    torch.save(model.classifier.state_dict(), f'{dir}/classifier.pt')

def save_config(config, dir):
    if not os.path.exists(dir): os.makedirs(dir)
    torch.save(config, f'{dir}/config.pt')

def load_model(model, dir):
    model.model = model.model.from_pretrained(dir)
    model.processor = model.processor.from_pretrained(dir)
    model.classifier.load_state_dict(torch.load(f'{dir}/classifier.pt'))

def load_config(dir):
    return torch.load(f'{dir}/config.pt')

save_dir = 'saved'