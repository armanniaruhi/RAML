# https://www.kaggle.com/datasets/kushsheth/face-vae/data
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data import Subset
import random
import torch


def _load_partitions(partition_file):
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


class CelebALabeledDataset(Dataset):
    def __init__(self, image_dir, label_file, img_size=64, transform=None, partition_file=None, partition_id=None, loss_type='contrastive'):
        """
        Args:
            image_dir (str): Path to the directory containing images.
            label_file (str): Path to the .txt file with image labels.
            img_size (int): Resize all images to this size.
            transform (callable, optional): Optional transforms to apply to images.
            partition_file (str, optional): Path to the partition file for train/val/test split.
            partition_id (int, optional): Partition ID to use (0=train, 1=val, 2=test).
        """
        self.loss_type = loss_type
        self.image_dir = image_dir
        self.label_map = self._load_labels(label_file)

        # Load partition information if provided
        if partition_file and partition_id is not None:
            partition_map = _load_partitions(partition_file)
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
                transforms.Resize([256, 256]),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        label = self.label_map[filename]
        img = Image.open(os.path.join(self.image_dir, filename)).convert('RGB')

        # Apply transformation
        if self.transform:
            img = self.transform(img)
        return img, label


def get_siamese_dataloader(image_dir, label_file, batch_size=32, img_size=64, shuffle=True, loss_type='contrastive'):
    """
    Creates and returns a DataLoader for Siamese network training

    Args:
        image_dir (str): Directory containing the images
        label_file (str): Path to the label file
        batch_size (int): Batch size for the dataloader
        img_size (int): Size to resize the images to
        shuffle (bool): Whether to shuffle the data
        loss_type (str): Type of loss function to use ('contrastive' or 'ms')

    Returns:
        DataLoader: PyTorch DataLoader object
    """
    dataset = CelebALabeledDataset(image_dir, label_file, img_size=img_size, loss_type=loss_type)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_partitioned_dataloaders(image_dir, label_file, partition_file, batch_size=32, img_size=64, loss_type='contrastive'):
    """
    Creates separate dataloaders for train, validation, and test sets based on partition file.

    Args:
        image_dir (str): Directory containing the images
        label_file (str): Path to the label file
        partition_file (str): Path to the list_eval_partition.csv file
        batch_size (int): Batch size for the dataloaders
        img_size (int): Size to resize the images to
        loss_type (str): Type of loss function to use ('contrastive' or 'ms')

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create datasets for each partition
    train_dataset = CelebALabeledDataset(image_dir, label_file, img_size=img_size,
                                         partition_file=partition_file, partition_id=0, loss_type=loss_type)
    val_dataset = CelebALabeledDataset(image_dir, label_file, img_size=img_size,
                                       partition_file=partition_file, partition_id=1, loss_type=loss_type)
    test_dataset = CelebALabeledDataset(image_dir, label_file, img_size=img_size,
                                        partition_file=partition_file, partition_id=2, loss_type=loss_type)

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
