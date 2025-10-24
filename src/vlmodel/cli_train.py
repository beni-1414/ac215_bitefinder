import torch
import torch.optim as optim
from trainer import Trainer
from model import *
from dataset import BugBitePairedDataset
from model import *
from utils_dataloader import *
from utils_save import *
import argparse

# Define training arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=list(model_classes.keys()), default='clip')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--device', type=str)
parser.add_argument('-s', '--save', action='store_true')
parser.add_argument('-v', '--verbose', action='store_true')

# Parse training arguments
args = parser.parse_args()
model_id = args.model
num_epochs = args.epochs
batch_size = args.batch_size
lr = args.lr
if args.device: device = torch.device(args.device)
else: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save = args.save
print(save)
verbose = args.verbose
gcp = False
seed = 42

dataset = BugBitePairedDataset(on_gcp=gcp, seed=seed)
model_class = model_classes[model_id]
model = model_class(num_labels=dataset.num_labels, freeze_params=True)
config = {
    'model_id': model_id,
    'label_to_id': dataset.label_to_id,
    'id_to_label': dataset.id_to_label,
    'num_labels': dataset.num_labels
}
processor = model.processor
optim_class = optim.Adam

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
)
trainer.train_eval()

if save:
    save_config(config, dir=save_dir)
    save_model(model, dir=save_dir)