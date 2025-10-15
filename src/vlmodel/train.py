import torch
import torch.optim as optim
from trainer import Trainer
from model import CLIPForBugBiteClassification, ViLTForBugBiteClassification
from dataset import BugBitePairedDataset, train_eval_split, collate_paired_fn
from save_utils import *

seed = 42

dataset = BugBitePairedDataset(seed=seed)
train_dataset, eval_dataset = train_eval_split(dataset, train_split=0.8, seed=seed)
model = CLIPForBugBiteClassification(num_labels=dataset.num_labels, freeze_params=False, dropout_prob=0.1)
model_class = 'clip'
config = {
    'model_class': model_class,
    'label_to_id': dataset.label_to_id,
    'id_to_label': dataset.id_to_label,
    'num_labels': dataset.num_labels
}
processor = model.processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save = True

num_epochs = 5
dataloader_kwargs = {
    'batch_size': 64, 
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
    device=device,
    verbose=True,
)
trainer.train()
trainer.eval()

if save:
    save_config(config, save_dir)
    save_model(model, save_dir)