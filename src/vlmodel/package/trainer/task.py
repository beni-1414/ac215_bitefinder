"""
Main training task for Google Cloud Vertex AI Custom Training.
This serves as the entry point for serverless training jobs.
"""
import os
import subprocess
import argparse
import torch
import torch.optim as optim

# Import from package modules
from trainer.trainer_module import Trainer
from trainer.model import model_classes
from trainer.dataset import BugBitePairedDataset


def main():
    """Main training function for serverless execution."""
    # Define training arguments
    parser = argparse.ArgumentParser(description='Train vision-language model for bug bite classification')
    parser.add_argument('--model', type=str, choices=list(model_classes.keys()), default='clip',
                        help='Model type to train (clip or vilt)')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu). Auto-detected if not specified')
    parser.add_argument('--labels', type=str, default='texts_v1.json',
                        help='Path to the training labels JSON file')
    parser.add_argument('--data_root_dir', type=str, default='data/',
                        help='Root directory containing training data')
    parser.add_argument('--output_dir', type=str, default='trained_model',
                        help='Directory to save trained model')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--gcp', action='store_true',
                        help='Flag to indicate data is on GCP (will download from bucket)')
    parser.add_argument('--run_id', type=str, default='default_run',
                        help='Identifier for the training run')
    
    # Parse training arguments
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🚀 Starting training job...")
    print(f"   Model: {args.model}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Device: {device}")
    print(f"   GCP mode: {args.gcp}")
    

    # Download dataset from bucket
    gcs_data_dir = os.getenv('GCS_BUCKET_URI') + '/' + args.data_root_dir
    local_data_dir = args.data_root_dir
    print(f"\n☁️ Downloading {gcs_data_dir} -> {local_data_dir}...")
    subprocess.run(["gsutil", "-m", "cp", "-r", gcs_data_dir, "."], check=True)
    print("✅ Download complete!")

    # Load dataset
    print(f"\n📊 Loading dataset...")
    dataset = BugBitePairedDataset(
        data_root_dir=local_data_dir,
        seed=args.seed,
        text_fname=args.labels
    )
    print(f"   Total samples: {len(dataset)}")
    print(f"   Number of classes: {dataset.num_labels}")
    
    # Initialize model
    print(f"\n🤖 Initializing {args.model.upper()} model...")
    model_class = model_classes[args.model]
    model = model_class(num_labels=dataset.num_labels, freeze_params=True)
    
    # Save configuration
    config = {
        'model_id': args.model,
        'label_to_id': dataset.label_to_id,
        'id_to_label': dataset.id_to_label,
        'num_labels': dataset.num_labels
    }
    
    # Initialize optimizer
    optimizer_class = optim.Adam
    
    # Create trainer
    print(f"\n🏋️ Setting up trainer...")
    trainer = Trainer(
        dataset=dataset,
        model=model,
        model_id=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer_class=optimizer_class,
        lr=args.lr,
        device=device,
        seed=args.seed,
        verbose=args.verbose,
        run_id=args.run_id
    )
    
    # Train and evaluate
    print(f"\n🎯 Starting training...")
    trainer.train_eval()
    
    # # Save model and configuration
    # print(f"\n💾 Saving model to {args.output_dir}...")
    # from trainer.utils_save import save_config, save_model
    # save_config(config, dir=args.output_dir, to_gcp=args.gcp, run_id=args.run_id)
    # save_model(model, dir=args.output_dir, to_gcp=args.gcp, run_id=args.run_id)

    print(f"\n✅ Training completed successfully!")


if __name__ == '__main__':
    main()
