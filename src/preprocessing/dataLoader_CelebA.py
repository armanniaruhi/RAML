# https://www.kaggle.com/datasets/kushsheth/face-vae/data
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import random
import torch
# "test"

class CelebALabeledDataset(Dataset):
    def __init__(self, image_dir, label_file, img_size=64, transform=None):
        """
        Args:
            image_dir (str): Path to the directory containing images.
            label_file (str): Path to the .txt file with image labels.
            img_size (int): Resize all images to this size.
            transform (callable, optional): Optional transforms to apply to images.
        """
        self.image_dir = image_dir
        self.label_map = self._load_labels(label_file)
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
            # Get another image from the same class
            positive_files = self.label_to_images[anchor_label]
            second_filename = random.choice([f for f in positive_files if f != anchor_filename])
            target = 1
        else:
            # Get an image from a different class
            other_labels = [l for l in self.label_to_images.keys() if l != anchor_label]
            other_label = random.choice(other_labels)
            second_filename = random.choice(self.label_to_images[other_label])
            target = 0

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
