import torch
import torch.optim as optim
from trainer import Trainer
from model import CLIPForBugBiteClassification, ViLTForBugBiteClassification
from dataset import BugBitePairedDataset, train_eval_split, collate_paired_fn

seed = 42

dataset = BugBitePairedDataset(seed=seed)
model = ViLTForBugBiteClassification(num_labels=dataset.num_labels, dropout_prob=0.1)
processor = model.processor
train_dataset, eval_dataset = train_eval_split(dataset, train_split=0.8, seed=seed)

num_epochs = 1
dataloader_kwargs = {
    'batch_size': 4, 
    'shuffle': True, 
    'collate_fn': lambda batch: collate_paired_fn(batch, processor)
}
optimizer_kwargs = {
    'lr': 1e-4
}

trainer = Trainer(
    train_dataset=train_dataset, 
    eval_dataset=eval_dataset, 
    model=model, 
    num_epochs=num_epochs, 
    dataloader_kwargs=dataloader_kwargs, 
    optimizer_class=optim.Adam, 
    optimizer_kwargs=optimizer_kwargs, 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    verbose=True,
)
trainer.train()
trainer.eval()