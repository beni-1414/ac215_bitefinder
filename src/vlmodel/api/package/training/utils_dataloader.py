import torch
from torch.utils.data import Dataset, random_split

'''
collate_paired_fn: collate function to process a batch of image-text pairs using a processor
- batch: list of items from BugBitePairedDataset
- processor: a processor for preprocessing image-text pairs
'''


def collate_paired_fn(batch, processor):
    # Stack text and image pairs from batch
    text = [item['text'] for item in batch]
    images = [item.get('image') for item in batch]
    # Pass inputs through processor to tokenize text and resize and normalize images, and convert inputs to tensors
    processed = processor(text=text, images=images, return_tensors='pt', padding=True)
    # Stack labels from batch into a single tensor
    processed['labels'] = torch.tensor([item['labels'] for item in batch])
    return processed


'''
train_eval_split: utility function to split a dataset into training and evaluation sets
- dataset: the full dataset
- train_split: proportion of dataset to use for training (rest used for evaluation)
- seed: random seed for reproducibility
'''


def train_eval_split(dataset: Dataset, train_split=0.8, seed=None):
    train_size = int(train_split * len(dataset))
    eval_size = len(dataset) - train_size
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
        return random_split(dataset, [train_size, eval_size], generator)
    else:
        return random_split(dataset, [train_size, eval_size])
