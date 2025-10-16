import torch
import torch.optim as optim
from trainer import Trainer
from model import CLIPForBugBiteClassification, ViLTForBugBiteClassification
from dataset import BugBitePairedDataset, train_eval_split, collate_paired_fn
from utils_model import model_class
from utils_save import *

seed = 42
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gcp = True
save = True

dataset = BugBitePairedDataset(on_gcp=gcp, seed=seed)
train_split = 0.8
train_dataset, eval_dataset = train_eval_split(dataset, train_split=train_split, seed=seed)
model = CLIPForBugBiteClassification(num_labels=dataset.num_labels, freeze_params=False)
config = {
    'model_class': model_class(model),
    'label_to_id': dataset.label_to_id,
    'id_to_label': dataset.id_to_label,
    'num_labels': dataset.num_labels
}
processor = model.processor

num_epochs = 1
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
    save_config(config, dir=save_dir)
    save_model(model, dir=save_dir)