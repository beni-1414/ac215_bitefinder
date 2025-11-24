import os
import argparse
import pkgutil
import yaml
import wandb
from training.trainer import Trainer
from training.model import model_classes, activation_funcs
from training.trainer import optim_classes
from training.dataset import BugBitePairedDataset
from training.utils_wandb import wandb_login
from training.utils_gcs import download_from_gcs
from datetime import datetime


def main():
    # Define training arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=list(model_classes.keys()), default='clip')
    parser.add_argument('--unfreeze_layers', type=int, default=2)
    parser.add_argument('--classifier_layers', type=int, default=1)
    parser.add_argument('--activation', type=str, choices=list(activation_funcs.keys()), default='relu')
    parser.add_argument('--dropout_prob', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--optimizer', type=str, choices=list(optim_classes.keys()), default='adam')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--text_fname', type=str, default='texts_v1.json')
    parser.add_argument('--data_root_dir', type=str, default='data/')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sweep_config', type=str, default=None)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-s', '--save', action='store_true')

    # Parse training arguments
    args = parser.parse_args()

    # Necessary setup pre-training
    pretraining(args)

    # Either run W&B sweep or single training job
    sweep_config_file = args.sweep_config
    if sweep_config_file:
        # Load sweep config
        sweep_config_dict = yaml.safe_load(pkgutil.get_data('training', sweep_config_file))
        # Create sweep
        sweep_id = wandb.sweep(sweep_config_dict, project=os.environ['WANDB_PROJECT'], entity=os.environ['WANDB_TEAM'])
        # Load static parameters from args not controlled by sweep
        sweep_params = set(sweep_config_dict.get('parameters', {}).keys())
        static_config = {}
        for key, value in vars(args).items():
            if key not in sweep_params:
                static_config[key] = value
        # Start W&B sweep agent (passing static parameters, sweep parameters automatically handled by W&B)
        print(f'🚀 Starting W&B sweep agent: {sweep_id}')
        wandb.agent(sweep_id, function=lambda: training(config=static_config))
        print('✅ Sweep job complete!')
    else:
        # Run training job with args (specified in job config or default value)
        print('🚀 Starting training job...')
        training(config=args)
        print('✅ Job complete!')


'''
pretraining: function to run necessary setup before training
'''


def pretraining(args):
    # Log into W&B
    wandb_login()

    # Download dataset from GCS bucket
    download_from_gcs(args.data_root_dir)


'''
training: function to be called by W&B sweep agent or standalone to run single training job; responsible for running one W&B training run:
instansiates dataset, model, and trainer, and runs training/validation loop
- config: training configuration arguments
    - for single training job, args passed from job config (command line args or defaults)
    - for W&B sweep job, static args passed from job config (not controlled by sweep, sweep args automatically handled by W&B)
'''


def training(config=None):
    with wandb.init(entity=os.environ['WANDB_TEAM'], project=os.environ['WANDB_PROJECT'], config=config):
        config = wandb.config  # Sweep parameters override where defined

        # Generate unique run name
        datetime_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f'{config.model}_{datetime_str}'
        wandb.run.name = run_name

        # Parse config arguments
        model_id = config.model
        unfreeze_layers = config.unfreeze_layers
        classifier_layers = config.classifier_layers
        activation = config.activation
        dropout_prob = config.dropout_prob
        num_epochs = config.epochs
        batch_size = config.batch_size
        optimizer_id = config.optimizer
        lr = config.lr
        weight_decay = config.weight_decay
        text_fname = config.text_fname
        seed = config.seed
        data_root_dir = config.data_root_dir
        verbose = config.verbose
        save = config.save

        # Initialize dataset
        dataset = BugBitePairedDataset(
            data_root_dir=data_root_dir,
            image_root_dir='images/',
            text_root_dir='text/',
            data_split='training',
            text_fname=text_fname,
            seed=seed,
        )

        # Initialize model
        model_class = model_classes[model_id]
        model = model_class(
            num_labels=dataset.num_labels,
            unfreeze_layers=unfreeze_layers,
            classifier_layers=classifier_layers,
            activation=activation,
            dropout_prob=dropout_prob,
        )

        # Initialize trainer
        trainer = Trainer(
            dataset=dataset,
            model=model,
            model_id=model_id,
            optimizer_id=optimizer_id,
            num_epochs=num_epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            seed=seed,
            verbose=verbose,
            save=save,
            save_dir='models',
        )

        # Update W&B config with additional parameters
        wandb.config.pretrained = model.pretrained
        wandb.config.total_params = trainer.total_params
        wandb.config.trainable_params = trainer.trainable_params
        wandb.config.dataset = dataset.dataset_id
        wandb.config.num_labels = dataset.num_labels
        wandb.config.id_to_label = dataset.id_to_label

        # Run training/validation loop
        trainer.train_eval()


if __name__ == '__main__':
    main()
