import os
import subprocess
import argparse
import torch
import torch.optim as optim
import wandb
from training.trainer import Trainer
from training.model import model_classes
from training.dataset import BugBitePairedDataset
from training.utils_gcp import get_secret

training_args = None


def main():
    global training_args

    # Define and parse training arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=list(model_classes.keys()), default='clip')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--text_fname', type=str, default='texts_v1.json')
    parser.add_argument('--data_root_dir', type=str, default='data/')
    parser.add_argument('--run_id', type=str, default='default_run')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sweep_id', type=str, default=None)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-s', '--save', action='store_true')
    training_args = parser.parse_args()

    # Log into W&B
    wandb_key = get_secret('WANDB_API_KEY')
    if wandb_key:
        os.environ['WANDB_API_KEY'] = wandb_key
    wandb.login()

    sweep_id = training_args.sweep_id
    if sweep_id:
        print(f'🚀 Starting W&B sweep agent: {sweep_id}')
        wandb.agent(sweep_id, function=run_training)
        print('✅ Sweep job complete!')
    else:
        print('🚀 Starting training job...')
        run_training()
        print('✅ Job complete!')


'''
run_training: training function to be called by W&B sweep agent or standalone to run single training job
'''


def run_training():
    args = training_args
    cfg = wandb.config if wandb.run else vars(args)
    model_id = cfg.get('model', args.model)
    num_epochs = cfg.get('epochs', args.epochs)
    batch_size = cfg.get('batch_size', args.batch_size)
    lr = cfg.get('lr', args.lr)
    text_fname = cfg.get('text_fname', args.text_fname)
    seed = cfg.get('seed', args.seed)
    run_id = args.run_id
    if wandb.run:
        run_id = f"{args.run_id}_{wandb.run.id}"
    data_root_dir = args.data_root_dir
    device = torch.device(args.device) if args.device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    verbose = args.verbose
    save = args.save

    # Download dataset from GCS bucket
    gcs_data_dir = os.getenv('GCS_BUCKET_URI') + '/' + data_root_dir
    local_data_dir = data_root_dir
    print(f'☁️ Downloading data from {gcs_data_dir}...')
    subprocess.run(['gsutil', '-q', '-m', 'cp', '-r', gcs_data_dir, '.'], check=True)
    print('✅ Download complete!')

    # Initialize dataset
    dataset = BugBitePairedDataset(
        data_root_dir=local_data_dir,
        image_root_dir='images/',
        text_root_dir='text/',
        data_split='training',
        text_fname=text_fname,
        seed=seed,
    )

    # Initialize model
    model_class = model_classes[model_id]
    model = model_class(num_labels=dataset.num_labels, freeze_params=True)
    optim_class = optim.Adam

    # Initialize trainer
    trainer = Trainer(
        dataset=dataset,
        model=model,
        model_id=model_id,
        num_epochs=num_epochs,
        batch_size=batch_size,
        optimizer_class=optim_class,
        lr=lr,
        device=device,
        seed=seed,
        verbose=verbose,
        run_id=run_id,
        save=save,
        save_dir='models',
    )

    # Run training/validation loop
    if verbose:
        print('🤖 Starting training...')
    trainer.train_eval()
    if verbose:
        print('✅ Training complete!')


if __name__ == '__main__':
    main()
