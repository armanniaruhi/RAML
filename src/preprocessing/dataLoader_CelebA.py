import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import torch
from pytorch_metric_learning.samplers import MPerClassSampler


def _load_partitions(partition_file):
    """Load train/val/test partition mappings from file."""
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
    def __init__(self, image_dir, label_file, img_size=64, transform=None,
                 partition_file=None, partition_id=None):
        """
        Args:
            image_dir (str): Path to image directory
            label_file (str): Path to label file
            img_size (int): Target image size
            transform (callable): Optional transforms
            partition_file (str): Path to partition file
            partition_id (int): 0=train, 1=val, 2=test
        """
        self.image_dir = image_dir
        self.label_map = self._load_labels(label_file)

        # Load partition information
        if partition_file and partition_id is not None:
            partition_map = _load_partitions(partition_file)
            self.image_files = [
                f for f in os.listdir(image_dir)
                if f in self.label_map and f in partition_map
                   and partition_map[f] == partition_id
            ]
        else:
            self.image_files = [f for f in os.listdir(image_dir) if f in self.label_map]

        # Store labels as list for sampler access
        self.labels = [self.label_map[img_file] for img_file in self.image_files]
        self.unique_labels = torch.unique(torch.tensor(self.labels)).tolist()

        # Default transforms if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize([256, 256]),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def _load_labels(self, label_file):
        """Load label mappings from file."""
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

        if self.transform:
            img = self.transform(img)
        return img, label


def get_dataloader(image_dir, label_file, batch_size=32, m_per_sample=2):
    """
    Creates DataLoader for contrastive learning with guaranteed positive pairs.
    Args:
        image_dir: Directory containing images
        label_file: Path to label file
        batch_size: Batch size
    Returns:
        DataLoader with MPerClassSampler
    """
    dataset = CelebALabeledDataset(image_dir, label_file)

    sampler = MPerClassSampler(
        labels=dataset.labels,
        m=m_per_sample,  # At least 2 samples per class per batch
        batch_size=batch_size,
        length_before_new_iter=len(dataset)
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True
    )


def get_partitioned_dataloaders(image_dir, label_file, partition_file, batch_size=32, m_per_sample=2):
    """
    Creates train/val/test dataloaders with proper sampling for contrastive learning.
    Args:
        image_dir: Image directory
        label_file: Label file path
        partition_file: Partition file path
        batch_size: Batch size
        img_size: Image size
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """

    train_dataset = CelebALabeledDataset( image_dir, label_file,
        partition_file=partition_file, partition_id=0)
    val_dataset = CelebALabeledDataset(
        image_dir, label_file,
        partition_file=partition_file, partition_id=1
    )
    test_dataset = CelebALabeledDataset(
        image_dir, label_file,
        partition_file=partition_file, partition_id=2
    )

    train_sampler = MPerClassSampler(
        labels=train_dataset.labels,
        m=m_per_sample,
        batch_size=batch_size,
        length_before_new_iter=len(train_dataset))

    val_sampler = MPerClassSampler(
        labels=val_dataset.labels,
        m=m_per_sample,
        batch_size=batch_size,
        length_before_new_iter=len(val_dataset)
    )

    test_sampler = MPerClassSampler(
        labels=test_dataset.labels,
        m=m_per_sample,
        batch_size=batch_size,
        length_before_new_iter=len(test_dataset)
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size = batch_size, sampler = train_sampler, drop_last = True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_sampler, shuffle=False)

    return train_loader, val_loader, test_loader


def create_subset_loader(original_loader, num_samples=1000):
    """Create a subset data loader from an existing loader."""
    original_dataset = original_loader.dataset
    subset_indices = torch.randperm(len(original_dataset))[:num_samples]
    subset_dataset = Subset(original_dataset, subset_indices)

    return DataLoader(
        subset_dataset,
        batch_size=original_loader.batch_size,
        shuffle=True,
        num_workers=original_loader.num_workers,
        pin_memory=original_loader.pin_memory
    )