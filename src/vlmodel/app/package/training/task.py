import os
import subprocess
import argparse
import torch
import torch.optim as optim

from training.trainer import Trainer
from training.model import model_classes
from training.dataset import BugBitePairedDataset

def main():
    print(f"🚀 Starting training job...")

    # Define training arguments
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
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-s', '--save', action='store_true')
    
    # Parse training arguments
    args = parser.parse_args()
    model_id = args.model
    num_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    if args.device: device = torch.device(args.device)
    else: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    text_fname = args.text_fname
    data_root_dir = args.data_root_dir
    run_id = args.run_id
    seed = args.seed
    verbose = args.verbose
    save = args.save

    # Download dataset from GCS bucket
    gcs_data_dir = os.getenv('GCS_BUCKET_URI') + '/' + data_root_dir
    local_data_dir = data_root_dir
    print(f"☁️ Downloading data from {gcs_data_dir}...")
    subprocess.run(["gsutil", "-q", "-m", "cp", "-r", gcs_data_dir, "."], check=True)
    print("✅ Download complete!")

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
    print(f"🤖 Starting training...")
    trainer.train_eval()
    print(f"✅ Training complete!")

if __name__ == '__main__':
    main()
