# https://www.kaggle.com/datasets/kushsheth/face-vae/data
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import Subset
import random
import torch

class CelebALabeledDataset(Dataset):
    def __init__(self, image_dir, label_file, img_size=64, transform=None, partition_file=None, partition_id=None):
        """
        Args:
            image_dir (str): Path to the directory containing images.
            label_file (str): Path to the .txt file with image labels.
            img_size (int): Resize all images to this size.
            transform (callable, optional): Optional transforms to apply to images.
            partition_file (str, optional): Path to the partition file for train/val/test split.
            partition_id (int, optional): Partition ID to use (0=train, 1=val, 2=test).
        """
        self.image_dir = image_dir
        self.label_map = self._load_labels(label_file)
        
        # Load partition information if provided
        if partition_file and partition_id is not None:
            partition_map = self._load_partitions(partition_file)
            self.image_files = [f for f in os.listdir(image_dir) 
                              if f in self.label_map and f in partition_map 
                              and partition_map[f] == partition_id]
        else:
            self.image_files = [f for f in os.listdir(image_dir) if f in self.label_map]

        # Group images by label
        self.label_to_images = {}
        for img_file in self.image_files:
            label = self.label_map[img_file]
            if label not in self.label_to_images:
                self.label_to_images[label] = []
            self.label_to_images[label].append(img_file)

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transform

    def _load_labels(self, label_file):
        label_map = {}
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    filename, label = parts
                    label_map[filename] = int(label)
        return label_map

    def _load_partitions(self, partition_file):
        partition_map = {}
        with open(partition_file, 'r') as f:
            # Skip header if it exists
            if 'csv' in partition_file.lower():
                next(f, None)
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    filename, partition = parts
                    partition_map[filename] = int(partition)
        return partition_map

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Returns a tuple containing:
        - anchor image
        - second image (either same class or different class)
        - target (1 if same class, 0 if different class)
        """
        # Get anchor image
        anchor_filename = self.image_files[idx]
        anchor_label = self.label_map[anchor_filename]
        anchor_img = Image.open(os.path.join(self.image_dir, anchor_filename)).convert('RGB')

        # Randomly decide whether to get a positive or negative pair
        should_get_same_class = random.randint(0, 1)

        if should_get_same_class:
            # Get another image from the same class, excluding the anchor itself
            positive_files = self.label_to_images[anchor_label]
            same_class_files = [f for f in positive_files if f != anchor_filename]

            if same_class_files:
                second_filename = random.choice(same_class_files)
            else:
                # Fallback: force a different class instead
                should_get_same_class = False  # fallback to negative
                other_labels = [l for l in self.label_to_images.keys() if l != anchor_label]
                other_label = random.choice(other_labels)
                second_filename = random.choice(self.label_to_images[other_label])
                target = 0

        if not should_get_same_class:
            # Get an image from a different class
            other_labels = [l for l in self.label_to_images.keys() if l != anchor_label]
            other_label = random.choice(other_labels)
            second_filename = random.choice(self.label_to_images[other_label])
            target = 0

        else:
            target = 1

        second_img = Image.open(os.path.join(self.image_dir, second_filename)).convert('RGB')

        # Apply transformations
        if self.transform:
            anchor_img = self.transform(anchor_img)
            second_img = self.transform(second_img)

        return anchor_img, second_img, torch.FloatTensor([target])

def get_siamese_dataloader(image_dir, label_file, batch_size=32, img_size=64, shuffle=True):
    """
    Creates and returns a DataLoader for Siamese network training

    Args:
        image_dir (str): Directory containing the images
        label_file (str): Path to the label file
        batch_size (int): Batch size for the dataloader
        img_size (int): Size to resize the images to
        shuffle (bool): Whether to shuffle the data

    Returns:
        DataLoader: PyTorch DataLoader object
    """
    dataset = CelebALabeledDataset(image_dir, label_file, img_size=img_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_partitioned_dataloaders(image_dir, label_file, partition_file, batch_size=32, img_size=64):
    """
    Creates separate dataloaders for train, validation, and test sets based on partition file.

    Args:
        image_dir (str): Directory containing the images
        label_file (str): Path to the label file
        partition_file (str): Path to the list_eval_partition.csv file
        batch_size (int): Batch size for the dataloaders
        img_size (int): Size to resize the images to

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create datasets for each partition
    train_dataset = CelebALabeledDataset(image_dir, label_file, img_size=img_size,
                                       partition_file=partition_file, partition_id=0)
    val_dataset = CelebALabeledDataset(image_dir, label_file, img_size=img_size,
                                     partition_file=partition_file, partition_id=1)
    test_dataset = CelebALabeledDataset(image_dir, label_file, img_size=img_size,
                                      partition_file=partition_file, partition_id=2)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def create_subset_loader(original_loader, num_samples=1000):
    """Create a subset data loader from the original loader"""
    # Get the original dataset
    original_dataset = original_loader.dataset

    # Create indices for the subset
    subset_indices = torch.randperm(len(original_dataset))[:num_samples]

    # Create subset dataset
    subset_dataset = Subset(original_dataset, subset_indices)

    # Create new loader with same parameters but subset dataset
    subset_loader = torch.utils.data.DataLoader(
        subset_dataset,
        batch_size=original_loader.batch_size,
        shuffle=True,  # Typically want to shuffle for training
        num_workers=original_loader.num_workers,
        pin_memory=original_loader.pin_memory
    )
    return subset_loader