import random
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import json
from utils_gcp import *

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
- data_name: identifier for dataset instance
- data_root_dir: directory containing all data
- image_root_dir: subdirectory containing image data
- text_root_dir: subdirectory containing text data
- data_split: which dataset split to use ('training' or 'testing')
- text_fname: name of text file containing patient narratives for each label
- image_extens: tuple of allowed image file extensions
- on_gcp: whether data is stored on cloud (if so, will download from GCP bucket, otherwise will load from local filesystem)
- seed: random seed for reproducibility
'''
class BugBitePairedDataset(Dataset):
    def __init__(
            self, 
            data_root_dir='data/', 
            image_root_dir='images/',
            text_root_dir='text/',
            data_split='training', 
            text_fname='texts.json',
            image_extens=('.jpg', '.jpeg'),
            on_gcp=False,
            seed=None
    ):        
        # Dataset of (image filepath, text, label) tuples
        self.dataset = []
        self.dataset_id = data_root_dir[:-1]

        # Set seed for reproducibility in dataset creation
        if seed is not None: random.seed(seed)
        
        # Download data from GCP bucket
        if on_gcp: download_directory_from_gcp(data_root_dir)

        # Build image and text directory paths
        image_dir = data_root_dir+image_root_dir+data_split+'/'
        text_dir = data_root_dir+text_root_dir+data_split+'/'

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