from torch.utils.data import Dataset, DataLoader
import torch
import os
import torch.optim as optim
from torch import nn
from tqdm import tqdm
import wandb
from trainer.utils_wandb import *
from trainer.utils_dataloader import *
from trainer.utils_gcp import get_secret

'''
Trainer: experiment handler for model training and evaluation with integrated W&B logging
- dataset: labeled paired image-text dataset
- model: vision-language model with processor
- num_epochs: number of training epochs
- batch_size: batch size for data loader
- optimizer_class: optimizer class
- lr: learning rate for optimizer
- device: device to run training on (CPU/GPU)
- seed: random seed for reproducibility
- verbose: whether to print training/evaluation progress
'''
class Trainer():
    def __init__(
            self, 
            dataset: Dataset, 
            model: nn.Module, 
            model_id: str,
            num_epochs: int,
            batch_size: int,
            optimizer_class=optim.Adam,
            lr=1e-4,
            device='cpu',
            seed=None,
            verbose=False,
            run_id="default_run"
    ):
        self.model = model
        self.model_id = model_id
        self.processor = model.processor
        self.run_id = run_id
        
        self.total_params = sum(p.numel() for p in self.model.parameters()) # Count parameters to log model complexity
        self.trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.dataset = dataset
        train_dataset, eval_dataset = train_eval_split(self.dataset, train_split=0.8, seed=seed) # Split dataset into training and validation sets

        self.batch_size = batch_size
        self.dataloader_kwargs = { # Define dataloader arguments
            'batch_size': self.batch_size, 
            'shuffle': True, 
            'collate_fn': lambda batch: collate_paired_fn(batch, self.processor) # Custom collate function
        }
        self.train_dataloader = DataLoader(train_dataset, **self.dataloader_kwargs)
        self.eval_dataloader = DataLoader(eval_dataset, **self.dataloader_kwargs)
        
        self.lr = lr
        self.optimizer_kwargs = { # Define optimizer arguments
            'lr': self.lr
        }
        self.optimizer = optimizer_class(filter(lambda param: param.requires_grad, model.parameters()), **self.optimizer_kwargs)
        
        self.num_epochs = num_epochs

        self.device = device
        self.verbose = verbose

        # Log into Weights & Biases
        wandb_key = get_secret('WANDB_API_KEY')
        if wandb_key:
            os.environ["WANDB_API_KEY"] = wandb_key
        else:
            pass # automatically looks for WANDB_API_KEY environment variable, which must be preset)
        wandb.login()
    '''
    train_eval: model training and validation loop
    '''
    def train_eval(self):
        # Initialize a W&B run
        wandb.init(
            entity=wandb_team, # Set the team where your project will be logged
            project=wandb_project, # Set the project where this run will be logged
            settings=wandb.Settings(quiet=True),
            name=self.run_id,
            config = {
                'model': self.model_id,
                'pretrained': self.model.pretrained,
                'total_params': self.total_params,
                'trainable_params': self.trainable_params,
                'dataset': self.dataset.dataset_id,
                'synthetic_labels': self.dataset.synthetic_labels,
                'classes': self.dataset.num_labels,
                'batch_size': self.batch_size,
                'epochs': self.num_epochs,
                'optimizer': self.optimizer.__class__.__name__,
                'lr': self.lr,
            }
        )

        # Metrics
        train_loss = 0
        train_acc = 0
        eval_loss = 0
        eval_acc = 0
        
        self.model.to(self.device)
        for epoch in range(self.num_epochs):
            # Training
            self.model.train()
            train_total_loss = 0
            train_correct = 0
            train_total = 0
            for batch in tqdm(self.train_dataloader, desc='Training', disable=not self.verbose):
                batch = {key: value.to(self.device) for key, value in batch.items()}
                # Forward pass 
                outputs = self.model(**batch)
                loss = outputs['loss']
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # Accumulate loss
                train_total_loss += loss.item()
                # Compute accuracy
                logits = outputs['logits']
                y_pred = torch.argmax(logits, dim=-1)
                y = batch['labels']
                train_correct += (y_pred == y).sum().item()
                train_total += y.size(0)
            # Compute and log epoch metrics
            train_loss = train_total_loss / len(self.train_dataloader)
            train_acc = train_correct / train_total
            wandb.log({"train_loss": train_loss, "train_accuracy": train_acc})
            if self.verbose: print(f'Epoch {epoch+1} | Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}')

            # Validation
            self.model.eval()
            with torch.no_grad(): # No gradient computation
                eval_total_loss = 0
                eval_correct = 0
                eval_total = 0
                for batch in tqdm(self.eval_dataloader, desc='Evaluating', disable=not self.verbose):
                    # Move tensors to device
                    batch = {key: value.to(self.device) for key, value in batch.items()}
                    # Forward pass
                    outputs = self.model(**batch)
                    loss = outputs['loss']
                    # Accumulate loss
                    eval_total_loss += loss.item()
                    # Compute accuracy
                    logits = outputs['logits']
                    y_pred = torch.argmax(logits, dim=-1)
                    y = batch['labels']
                    eval_correct += (y_pred == y).sum().item()
                    eval_total += y.size(0)
                # Compute and log eval metrics
                eval_loss = eval_total_loss / len(self.eval_dataloader)
                eval_acc = eval_correct / eval_total
                wandb.log({"eval_loss": eval_loss, "eval_accuracy": eval_acc})
                if self.verbose: print(f'Loss: {eval_loss:.4f} | Accuracy: {eval_acc:.4f}')
        
        # Finish the W&B run
        wandb.run.finish()