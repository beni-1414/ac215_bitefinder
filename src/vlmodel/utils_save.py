import os
import torch
from utils_gcp import *

save_dir = 'trained_model'

def save_model(model, dir, to_gcp=False, run_id=''):
    if not os.path.exists(dir): os.makedirs(dir)
    model.model.save_pretrained(dir)
    model.processor.save_pretrained(dir)
    torch.save(model.classifier.state_dict(), f'{dir}/classifier.pt')
    if to_gcp: upload_directory_to_gcp(f'{dir}/model_{run_id}')

def save_config(config, dir, to_gcp=False, run_id=''):
    if not os.path.exists(dir): os.makedirs(dir)
    torch.save(config, f'{dir}/config.pt')
    if to_gcp: upload_file_to_bucket(f'{dir}/config_{run_id}.pt')

def load_model(model, dir, from_gcp=False, run_id=''):
    if from_gcp: download_directory_from_gcp(dir)
    model.model = model.model.from_pretrained(dir)
    model.processor = model.processor.from_pretrained(dir)
    model.classifier.load_state_dict(torch.load(f'{dir}/classifier_{run_id}.pt'))

def load_config(dir, from_gcp=False, run_id=''):
    if from_gcp: download_file_from_gcp(f'{dir}/config_{run_id}.pt')
    return torch.load(f'{dir}/config_{run_id}.pt')