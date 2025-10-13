from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm

'''
Trainer: handler for model training and evaluation
- train_dataset: training dataset
- eval_dataset: evaluation dataset
- model: vision-language model with processor
- num_epochs: number of training epochs
- dataloader_kwargs: arguments for data loader (e.g., batch_size, shuffle, collate_fn)
- optimizer_class: optimizer class
- optimizer_kwargs: arguments for optimizer (e.g., lr, weight_decay)
- device: device to run training on (CPU/GPU)
- verbose: whether to print training/evaluation progress
'''
class Trainer():
    def __init__(
            self, 
            train_dataset: Dataset, 
            eval_dataset: Dataset, 
            model: nn.Module, 
            num_epochs,
            dataloader_kwargs={},
            optimizer_class=optim.Adam,
            optimizer_kwargs={},
            device='cpu',
            verbose=False,
    ):
        self.model = model
        self.processor = model.processor
        self.train_dataloader = DataLoader(train_dataset, **dataloader_kwargs)
        self.eval_dataloader = DataLoader(eval_dataset, **dataloader_kwargs)
        self.optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        self.num_epochs = num_epochs
        self.device = device
        self.verbose = verbose

    '''
    train: model training loop
    - ret: whether to return final training metrics
    '''
    def train(self, ret=False):
        self.model.to(self.device)
        train_loss = 0
        train_acc = 0
        for epoch in range(self.num_epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            for batch in tqdm(self.train_dataloader, desc='Training'):
                batch = {key: value.to(self.device) for key, value in batch.items()}
                # Forward pass 
                outputs = self.model(**batch)
                loss = outputs['loss']
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # Accumulate loss
                total_loss += loss.item()
                # Compute accuracy
                logits = outputs['logits']
                y_pred = torch.argmax(logits, dim=-1)
                y = batch['labels']
                correct += (y_pred == y).sum().item()
                total += y.size(0)
            # Compute and display epoch metrics
            train_loss = total_loss / len(self.train_dataloader)
            train_acc = correct / total
            if self.verbose: print(f'Epoch {epoch+1} | Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}')
        if ret: return (train_loss, train_acc) # Optionally return final loss and accuracy

    '''
    eval: model evaluation
    - ret: whether to return final evaluation metrics
    '''
    def eval(self, ret=False):
        self.model.eval()
        with torch.no_grad(): # No gradient computation
            total_loss = 0
            correct = 0
            total = 0
            for batch in tqdm(self.eval_dataloader, desc='Evaluating'):
                # Move tensors to device
                batch = {key: value.to(self.device) for key, value in batch.items()}
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs['loss']
                # Accumulate loss
                total_loss += loss.item()
                # Compute accuracy
                logits = outputs['logits']
                y_pred = torch.argmax(logits, dim=-1)
                y = batch['labels']
                correct += (y_pred == y).sum().item()
                total += y.size(0)
            # Compute and display eval metrics
            eval_loss = total_loss / len(self.eval_dataloader)
            eval_acc = correct / total
            if self.verbose: print(f'Loss: {eval_loss:.4f} | Accuracy: {eval_acc:.4f}')
        if ret: return (eval_loss, eval_acc)