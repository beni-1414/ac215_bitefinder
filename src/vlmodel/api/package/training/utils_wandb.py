import wandb
import os
from training.utils_secret import get_secret


def wandb_login():
    wandb_key = get_secret('WANDB_API_KEY')
    if wandb_key:
        os.environ['WANDB_API_KEY'] = wandb_key
    wandb.login()
