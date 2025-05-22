import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# Fixed Synthetic Dataset Class
class SyntheticSiameseDataset(Dataset):
    def __init__(self, num_samples=1000, img_size=(3, 224, 224)):
        self.num_samples = num_samples
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Generate reproducible random pairs
        np.random.seed(42)
        self.pairs = []
        for _ in range(num_samples):
            if np.random.rand() > 0.5:
                # Similar pair (same class)
                base_img = np.random.rand(*img_size).astype(np.float32)
                img1 = base_img * 255
                img2 = (base_img + np.random.normal(0, 0.1, size=img_size).astype(np.float32)) * 255
                label = 1.0
            else:
                # Dissimilar pair (different class)
                img1 = np.random.rand(*img_size).astype(np.float32) * 255
                img2 = np.random.rand(*img_size).astype(np.float32) * 255
                label = -1.0
            self.pairs.append((img1, img2, label))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img1, img2, label = self.pairs[idx]

        # Convert to PIL Images first
        img1_pil = Image.fromarray(img1.transpose(1, 2, 0).astype('uint8'))
        img2_pil = Image.fromarray(img2.transpose(1, 2, 0).astype('uint8'))

        return (
            self.transform(img1_pil),
            self.transform(img2_pil),
            torch.tensor(label, dtype=torch.float32)
        )