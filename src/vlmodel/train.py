import torch
import torch.optim as optim
from trainer import Trainer
from model import *
from dataset import BugBitePairedDataset, train_eval_split, collate_paired_fn
from utils_model import model_classes
from utils_save import *

# Training arguments
model_id = 'clip'
train_split = 0.8
num_epochs = 5
batch_size = 64
lr = 1e-4
gcp = True
save = True

seed = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = BugBitePairedDataset(on_gcp=gcp, seed=seed)
train_dataset, eval_dataset = train_eval_split(dataset, train_split=train_split, seed=seed)
model_class = model_classes[model_id]
model = model_class(num_labels=dataset.num_labels, freeze_params=True)
config = {
    'model_id': model_id,
    'label_to_id': dataset.label_to_id,
    'id_to_label': dataset.id_to_label,
    'num_labels': dataset.num_labels
}
processor = model.processor

dataloader_kwargs = {
    'batch_size': batch_size, 
    'shuffle': True, 
    'collate_fn': lambda batch: collate_paired_fn(batch, processor)
}
optimizer_kwargs = {
    'lr': lr
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
    save_config(config, dir=save_dir)
    save_model(model, dir=save_dir)