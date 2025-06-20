import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import torch
from pytorch_metric_learning.samplers import MPerClassSampler
import random
from torch.utils.data import random_split, DataLoader

train_transform = transforms.Compose([
    transforms.Resize(128),  # Etwas höhere Auflösung für bessere Details
    transforms.RandomCrop(112),  # Standardgröße für Gesichtserkennungsmodelle
    transforms.RandomHorizontalFlip(p=0.5),

    # Geometrische Transformationen (vor Farboperationen)
    transforms.RandomApply([
        transforms.RandomAffine(
            degrees=7,
            translate=(0.03, 0.03),
            scale=(0.92, 1.08),
            shear=5
        )
    ], p=0.6),

    # Beleuchtungsvariationen
    transforms.RandomApply([
        transforms.ColorJitter(
            brightness=0.18,
            contrast=0.18,
            saturation=0  # Für Graustufen irrelevant
        )
    ], p=0.5),

    # Perspektivische Verzerrungen (simuliert Kamerawinkel)
    transforms.RandomPerspective(
        distortion_scale=0.12,
        p=0.35
    ),

    # Rauschen und Unschärfe
    transforms.RandomApply([transforms.GaussianBlur(
        kernel_size=(3, 3),
        sigma=(0.1, 0.8)
    )], p=0.3),

    transforms.RandomApply([transforms.GaussianNoise(
        mean=0.0,
        std=0.02
    )], p=0.25),

    # Qualitätsdegradation
    transforms.RandomApply([transforms.RandomResizedCrop(
        size=112,
        scale=(0.85, 0.95),
        ratio=(0.95, 1.05)
    ], p=0.4),

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),  # ImageNet-Statistiken für Graustufen
])

eval_transform = transforms.Compose([
    transforms.Resize([100, 100]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


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
    def __init__(self, image_dir, label_file, image_size=100, transform=None,
                 partition_file=None, partition_id=None, output_format='triplet'):
        """
        Args:class CelebALabeledDataset(Dataset):

            image_dir (str): Path to image directory
            label_file (str): Path to label file
            img_size (int): Target image size
            transform (callable): Optional transforms
            partition_file (str): Path to partition file
            partition_id (int): 0=train, 1=val, 2=test
            output_format (str): 'triplet' or 'siamese' (5-tuple)
        """
        self.image_dir = image_dir
        self.label_map = self._load_labels(label_file)
        self.output_format = output_format

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
        self.label_to_indices = {label: [] for label in self.unique_labels}
        
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)

        # Default transforms if none provided
        # Default transforms if none provided
        if transform is None:
            self.transform = train_transform if partition_id == 0 else eval_transform
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
        if self.output_format == 'triplet':
            return self._get_triplet_item(idx)
        else:  # siamese (5-tuple)
            return self._get_siamese_item(idx)

    def _get_triplet_item(self, idx):
        """Return anchor, positive, negative images for triplet loss"""
        anchor_img, anchor_label = self._load_single_item(idx)
        
        # Find positive sample
        positive_idx = idx
        while positive_idx == idx:  # Ensure different image
            positive_idx = random.choice(self.label_to_indices[anchor_label])
        positive_img, _ = self._load_single_item(positive_idx)
        
        # Find negative sample
        negative_label = random.choice([l for l in self.unique_labels if l != anchor_label])
        negative_idx = random.choice(self.label_to_indices[negative_label])
        negative_img, _ = self._load_single_item(negative_idx)
        
        return anchor_img, positive_img, negative_img, anchor_label

    def _get_siamese_item(self, idx):
        """Return img1, img2, similarity, label1, label2 (5-tuple)"""
        img0_tuple = (os.path.join(self.image_dir, self.image_files[idx]), self.labels[idx])
        
        # 50% chance to get same class
        should_get_same_class = random.randint(0, 1)
        
        if should_get_same_class:
            while True:
                # Find same class image
                img1_idx = random.choice(range(len(self.image_files)))
                if self.labels[img1_idx] == img0_tuple[1] and img1_idx != idx:
                    break
        else:
            while True:
                # Find different class image
                img1_idx = random.choice(range(len(self.image_files)))
                if self.labels[img1_idx] != img0_tuple[1]:
                    break
        
        img1_tuple = (os.path.join(self.image_dir, self.image_files[img1_idx]), self.labels[img1_idx])
        
        try:
            #img0 = Image.open(img0_tuple[0]).convert('RGB')
            #img1 = Image.open(img1_tuple[0]).convert('RGB')
            img0 = Image.open(img0_tuple[0]).convert('L')
            img1 = Image.open(img1_tuple[0]).convert('L')
        except Exception as e:
            print(f"[Siamese] Fehler beim Öffnen von Bildern ({img0_tuple[0]} oder {img1_tuple[0]}): {e}")
            return self.__getitem__((idx + 1) % len(self.image_files))

        
        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            
        return (
            img0, 
            img1, 
            torch.tensor([int(img0_tuple[1] != img1_tuple[1])], dtype=torch.float32),
            img0_tuple[1],
            img1_tuple[1]
        )

    def _load_single_item(self, idx):
        """Helper to load a single image and label"""
        filename = self.image_files[idx]
        label = self.label_map[filename]
        try:
            #img = Image.open(os.path.join(self.image_dir, filename)).convert('RGB')
            img = Image.open(os.path.join(self.image_dir, filename)).convert('L')

        except Exception as e:
            print(f"[Triplet] Fehler beim Öffnen von {filename}: {e}")
            return self._load_single_item((idx + 1) % len(self.image_files))

        
        if self.transform:
            img = self.transform(img)
            
        return img, label


def get_partitioned_dataloaders(
    image_dir, label_file, partition_file, image_size=100,
    batch_size=32, m_per_sample=2, 
    num_identities=500, seed=42,
    output_format="siamese"
):
    """
    Creates train/val/test dataloaders from fixed-identity subsets with MPerClassSampler.
    Labels are remapped from original labels to consecutive integers starting at 0.
    """

    # Load the full dataset for train partition (partition_id=0) to get all labels
    full_dataset = CelebALabeledDataset(
        image_dir, label_file, image_size= image_size,
        partition_file=partition_file, partition_id=0,
        output_format=output_format
    )

    # Fix random seed for reproducibility
    random.seed(seed)
    torch.manual_seed(seed)

    # Select fixed subset of unique identities/labels
    all_labels = list(set(full_dataset.labels))
    selected_labels = set(random.sample(all_labels, min(num_identities, len(all_labels))))

    # Filter image files and labels to only include selected labels
    filtered_image_files = [
        fname for fname, label in zip(full_dataset.image_files, full_dataset.labels)
        if label in selected_labels
    ]

    filtered_label_map = {
        fname: full_dataset.label_map[fname]
        for fname in filtered_image_files
    }

    # Get original labels in filtered data and create remapping
    original_labels_filtered = [filtered_label_map[fname] for fname in filtered_image_files]
    unique_filtered_labels = sorted(set(original_labels_filtered))
    label_remap = {orig_label: new_label for new_label, orig_label in enumerate(unique_filtered_labels)}

    # Remap the labels in the filtered_label_map
    remapped_label_map = {fname: label_remap[label] for fname, label in filtered_label_map.items()}

    # Define a filtered dataset class with remapped labels
    class FilteredCelebADataset(CelebALabeledDataset):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.image_files = filtered_image_files
            self.label_map = remapped_label_map
            self.labels = [self.label_map[f] for f in self.image_files]
            self.unique_labels = torch.unique(torch.tensor(self.labels)).tolist()
            self.label_to_indices = {label: [] for label in self.unique_labels}
            for idx, label in enumerate(self.labels):
                self.label_to_indices[label].append(idx)

    # Create train dataset (partition_id=0) with filtered and remapped labels
    dataset = FilteredCelebADataset(
        image_dir, label_file,
        partition_file=partition_file, partition_id=0,
        output_format=output_format
    )
    
    dataset_length = len(dataset)
    train_size = int(0.7 * dataset_length)
    val_size = int(0.15 * dataset_length)
    test_size = dataset_length - train_size - val_size  # Ensure exact total

    # Ensure reproducibility
    generator = torch.Generator().manual_seed(seed)

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    # Create sampler only for training data
    train_sampler = MPerClassSampler(
        labels=[dataset.labels[i] for i in train_dataset.indices], 
        m=m_per_sample,
        batch_size=batch_size,
        length_before_new_iter=len(train_dataset)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        drop_last=True,
        num_workers=0
    )
    
    # For validation and test, use sequential samplers
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Important for evaluation
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0
    )

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