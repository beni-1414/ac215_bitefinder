from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb
from training.utils_dataloader import collate_paired_fn, train_eval_split

'''
Trainer: experiment handler for model training and evaluation with integrated W&B logging
- dataset: labeled paired image-text dataset
- model: vision-language model with processor
- num_epochs: number of training epochs
- batch_size: batch size for data loader
- optimizer_id: optimizer class
- lr: learning rate for optimizer
- weight_decay: weight decay for optimizer
- device: device to run training on (CPU/GPU)
- seed: random seed for reproducibility
- verbose: whether to print training/evaluation progress
- save: whether to save the model as a W&B artifact
'''


class Trainer:
    def __init__(
        self,
        dataset: Dataset,
        model: nn.Module,
        model_id: str,
        optimizer_id: str,
        num_epochs: int,
        batch_size: int,
        lr=1e-4,
        weight_decay=0,
        seed=None,
        verbose=False,
        save=False,
        save_dir=None,
    ):
        self.model = model
        self.model_id = model_id
        self.processor = model.processor
        self.dataset = dataset
        self.seed = seed
        self.verbose = verbose
        self.num_epochs = num_epochs
        self.save = save
        self.save_dir = save_dir

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.total_params = sum(p.numel() for p in self.model.parameters())  # Count parameters to log model complexity
        self.trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # Split dataset into training and validation sets
        train_dataset, eval_dataset = train_eval_split(self.dataset, train_split=0.8, seed=self.seed)

        self.batch_size = batch_size
        self.dataloader_kwargs = {  # Define dataloader arguments
            'batch_size': self.batch_size,
            'shuffle': True,
            'collate_fn': lambda batch: collate_paired_fn(batch, self.processor),  # Custom collate function
        }
        self.train_dataloader = DataLoader(train_dataset, **self.dataloader_kwargs)
        self.eval_dataloader = DataLoader(eval_dataset, **self.dataloader_kwargs)

        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_kwargs = {'lr': self.lr, 'weight_decay': self.weight_decay}  # Define optimizer arguments
        optimizer_class = optim_classes[optimizer_id]
        self.optimizer = optimizer_class(filter(lambda param: param.requires_grad, model.parameters()), **self.optimizer_kwargs)

    '''
    train_eval: model training and validation loop
    '''

    def train_eval(self):
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
            wandb.log({'train_loss': train_loss, 'train_accuracy': train_acc})
            if self.verbose:
                print(f'Epoch {epoch+1} | Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}')

            # Validation
            self.model.eval()
            with torch.no_grad():  # No gradient computation
                eval_total_loss = 0
                eval_correct = 0
                eval_total = 0
                eval_logits = []
                eval_labels = []
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
                    eval_logits.append(logits.cpu())
                    eval_labels.append(y.cpu())
                # Compute and log eval metrics
                eval_loss = eval_total_loss / len(self.eval_dataloader)
                eval_acc = eval_correct / eval_total
                wandb.log({'eval_loss': eval_loss, 'eval_accuracy': eval_acc})
                if self.verbose:
                    print(f'Loss: {eval_loss:.4f} | Accuracy: {eval_acc:.4f}')
                # Compute confidence metrics
                logits_cat = torch.cat(eval_logits, dim=0)
                labels_cat = torch.cat(eval_labels, dim=0)
                probs = F.softmax(logits_cat, dim=1)
                confidences, predictions = probs.max(dim=1)
                correct = predictions.eq(labels_cat)
                confidence_metrics = {
                    'eval_avg_conf': confidences.mean().item(),
                    'eval_avg_corr_conf': confidences[correct].mean().item() if correct.any() else 0,
                }
                wandb.log(confidence_metrics)

        # Save trained model as W&B artifact
        if self.save:
            model_kwargs = {  # Model instansiation arguments
                'num_labels': self.dataset.num_labels,
                'unfreeze_layers': self.model.unfreeze_layers,
                'classifier_layers': self.model.classifier_layers,
                'dropout_prob': self.model.dropout_prob,
                'activation': self.model.activation,
            }
            artifact_metadata = {  # Store necessary metadata for future model inference
                'model_id': self.model_id,
                'model_kwargs': model_kwargs,
                'id_to_label': self.dataset.id_to_label,
            }
            artifact = wandb.Artifact(
                name=wandb.run.name,  # Use W&B run name to identify artifact
                type='model',
                metadata=artifact_metadata,
            )
            self.model.model.save_pretrained(self.save_dir)  # Save model and processor
            self.model.processor.save_pretrained(self.save_dir)
            torch.save(self.model.classifier.state_dict(), self.save_dir + '/classifier.pt')  # Save classification head
            artifact.add_dir(self.save_dir)
            wandb.log_artifact(artifact)


'''
optim_classes: map of optimizer name (in args and logs) to optimizer class
'''
optim_classes = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
}
