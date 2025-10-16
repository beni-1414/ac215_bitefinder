import random
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from torch.utils.data import random_split
import json
from utils_gcp import *
import io

'''
Assumes the following directory structure:
data/
    image/
        training/
            bug_1/
                img_1.jpg
                img_2.jpg
                ...
            bug_2/
            ...
            bug_n/
        testing/
            ...
    text/
        training/
            texts.json
        testing/
            texts.json
'''

'''
BugBitePairedDataset: labeled paired image-text dataset for bug bite images + patient narratives
- processor: a processor for preprocessing image-text pairs
- data_root_dir: directory containing all data
- image_root_dir: subdirectory containing image data
- text_root_dir: subdirectory containing text data
- dataset: which dataset split to use ('training' or 'testing')
- text_fname: name of text file containing patient narratives for each label
- image_extens: tuple of allowed image file extensions
- on_cloud: whether data is stored on cloud (if so, will download from GCP bucket, otherwise will load from local system)
- seed: random seed for reproducibility
'''
class BugBitePairedDataset(Dataset):
    def __init__(
            self, 
            data_root_dir='data/', 
            image_root_dir='images/',
            text_root_dir='text/',
            dataset='training', 
            text_fname='texts.json',
            image_extens=('.jpg', '.jpeg'),
            on_gcp=False,
            seed=None
    ):
        # Dataset of (image filepath, text, label) tuples
        self.dataset = []
        
        # Set seed for reproducibility in dataset creation
        if seed is not None: random.seed(seed)
        
        # Download data from GCP bucket
        if on_gcp: download_directory_from_gcp(data_root_dir)

        # Build image and text directory paths
        image_dir = data_root_dir+image_root_dir+dataset+'/'
        text_dir = data_root_dir+text_root_dir+dataset+'/'

        # Get list of labels (which correspond to image subdirectories)
        labels = os.listdir(image_dir)

        # Encode labels to integers and vice versa
        self.label_to_id = {label: i for i, label in enumerate(sorted(labels))}
        self.id_to_label = {i: label for i, label in enumerate(sorted(labels))}
        self.num_labels = len(self.label_to_id)
        
        # Load texts JSON file
        with open(text_dir+text_fname, 'r') as text_file:
            text_data = json.load(text_file)
        
            # Build dataset of (image filepath, text, label) tuples
            for label in self.label_to_id:
                # Get list of text narratives for each label
                narratives = text_data[label]
                
                # Load list of image files for each label
                image_label_dir = image_dir+label+'/'
                image_files = os.listdir(image_label_dir)
                
                for image_file in image_files:
                    # Filter out non-image files
                    if not image_file.lower().endswith(image_extens): continue
                    # Construct image filepath
                    image_filepath = os.path.join(image_label_dir, image_file)
                    # Randomly select a narrative for each image
                    text = random.choice(narratives)
                    # Add (image filepath, text, label) tuple to dataset
                    self.dataset.append((image_filepath, text, label))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_fp, text, label = self.dataset[idx]
        # Load image
        image = Image.open(image_fp).convert('RGB')        
        # Return raw image and text (collate_fn will use processor to batch/pad) and encoded label
        item = {'image': image, 'text': text}
        item['labels'] = torch.tensor(self.label_to_id[label])
        return item

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