import random
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
from torch.utils.data import random_split
import json

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
    text.json
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
- seed: random seed for reproducibility
'''
class BugBitePairedDataset(Dataset):
    def __init__(
            self, 
            data_root_dir='data/', 
            image_root_dir='bug-bite-images/', 
            dataset='training', 
            text_fname='synthetic_text.json',
            image_extens=('.jpg', '.jpeg'),
            seed=None
    ):
        if seed is not None: random.seed(seed) # Set seed for reproducibility in dataset creation
        self.dataset = [] # Dataset of tuples: (image filepath, text, label)
        image_dir = data_root_dir+image_root_dir+dataset+'/' # Image dataset directory
        self.label_to_id = {label: i for i, label in enumerate(sorted(os.listdir(image_dir)))} # Encoded label map
        self.id_to_label = {i: label for i, label in enumerate(sorted(os.listdir(image_dir)))} # Decoded label map
        self.num_labels = len(self.label_to_id) # Number of classes
        with open(data_root_dir+text_fname, 'r') as text_file: # Parse text JSON file
            text_data = json.load(text_file)
            for label in self.label_to_id:
                narratives = text_data[label] # Get list of text narratives for each label
                image_label_dir = image_dir+label+'/'
                # Load list of image filepaths for each label
                for image_fname in os.listdir(image_label_dir):
                    if not image_fname.lower().endswith(image_extens): continue # Filter out non-image files
                    image_fliepath = os.path.join(image_label_dir, image_fname)
                    text = random.choice(narratives) # Randomly select a narrative for each image
                    self.dataset.append((image_fliepath, text, label)) # Add (image filepath, text, label) tuple to dataset
        

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_file, text, label = self.dataset[idx]
        image = Image.open(image_file).convert('RGB') # Load PIL image from filepath
        item = {'image': image, 'text': text} # Return raw image and text (collate_fn will use processor to batch/pad)
        item['labels'] = torch.tensor(self.label_to_id[label]) # Encode label to integer
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
    
def read_gcp():
    # # Initiate Storage client
    # storage_client = storage.Client(project=gcp_project)

    # # Get reference to bucket
    # bucket = storage_client.bucket(bucket_name)

    # # Find all content in a bucket
    # blobs = bucket.list_blobs(prefix="input_audios/")
    # for blob in blobs:
    #     print(blob.name)
    #     if not blob.name.endswith("/"):
    #         blob.download_to_filename(blob.name)
    pass