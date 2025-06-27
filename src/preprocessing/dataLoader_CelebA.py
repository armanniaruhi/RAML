import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import torch
from pytorch_metric_learning.samplers import MPerClassSampler
import random
from torch.utils.data import random_split, DataLoader
from PIL import Image
import numpy as np


from PIL import Image, ImageFilter, ImageEnhance

## Custom augmentations
class CenterZoom:
    def __init__(self, zoom_factor=1.5):
        self.zoom_factor = zoom_factor

    def __call__(self, img):
        width, height = img.size
        new_width = int(width / self.zoom_factor)
        new_height = int(height / self.zoom_factor)
        left = (width - new_width) // 2
        top = (height - new_height) // 2
        right = left + new_width
        bottom = top + new_height
        img = img.crop((left, top, right, bottom))
        return img.resize((width, height))


class RandomRotate:
    def __init__(self, degrees=15):
        self.degrees = degrees

    def __call__(self, img):
        angle = random.uniform(-self.degrees, self.degrees)
        return img.rotate(angle, resample=Image.BILINEAR, expand=False)


class RandomBlur:
    def __init__(self, max_radius=2):
        self.max_radius = max_radius

    def __call__(self, img):
        radius = random.uniform(0, self.max_radius)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))


class RandomBrightnessContrast:
    def __init__(self, brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2)):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(self, img):
        enhancer = ImageEnhance.Brightness(img)
        brightness_factor = random.uniform(*self.brightness_range)
        img = enhancer.enhance(brightness_factor)

        enhancer = ImageEnhance.Contrast(img)
        contrast_factor = random.uniform(*self.contrast_range)
        img = enhancer.enhance(contrast_factor)
        return img


class RandomNoise:
    def __init__(self, noise_level=0.05):
        self.noise_level = noise_level

    def __call__(self, img):
        img_np = np.array(img) / 255.0
        noise = np.random.normal(0, self.noise_level, img_np.shape)
        noisy_img = img_np + noise
        noisy_img = np.clip(noisy_img, 0, 1)
        noisy_img = (noisy_img * 255).astype(np.uint8)
        return Image.fromarray(noisy_img)


train_transform = transforms.Compose([
    RandomRotate(degrees=10),  # Increased from 2
    CenterZoom(zoom_factor=1.5),  # Reduced from 1.5
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    CenterZoom(zoom_factor=1.5),
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def _load_partitions(partition_file):
    partition_map = {}
    with open(partition_file, 'r') as f:
        if 'csv' in partition_file.lower():
            next(f, None)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 2:
                filename, partition = parts
                partition_map[filename] = int(partition)
    return partition_map


class CelebALabeledDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None,
                 partition_file=None, partition_id=None,
                 to_rgb=True):
        """
        Args:
            image_dir (str): Path to image directory
            label_file (str): Path to label file
            transform (callable): Optional transforms
            partition_file (str): Path to partition file
            partition_id (int): 0=train, 1=val, 2=test
            to_rgb (bool): If True, convert images to RGB
        """
        self.image_dir = image_dir
        self.label_map = self._load_labels(label_file)
        self.to_rgb = to_rgb

        if partition_file and partition_id is not None:
            partition_map = _load_partitions(partition_file)
            self.image_files = [
                f for f in os.listdir(image_dir)
                if f in self.label_map and f in partition_map and partition_map[f] == partition_id
            ]
        else:
            self.image_files = [f for f in os.listdir(image_dir) if f in self.label_map]

        self.labels = [self.label_map[img_file] for img_file in self.image_files]
        self.unique_labels = torch.unique(torch.tensor(self.labels)).tolist()
        self.label_to_indices = {label: [] for label in self.unique_labels}
        for idx, label in enumerate(self.labels):
            self.label_to_indices[label].append(idx)

        self.transform = transform if transform else (train_transform if partition_id == 0 else eval_transform)

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
        return self._get_siamese_item(idx)

    def _get_triplet_item(self, idx):
        anchor_img, anchor_label = self._load_single_item(idx)

        positive_idx = idx
        while positive_idx == idx:
            positive_idx = random.choice(self.label_to_indices[anchor_label])
        positive_img, _ = self._load_single_item(positive_idx)

        negative_label = random.choice([l for l in self.unique_labels if l != anchor_label])
        negative_idx = random.choice(self.label_to_indices[negative_label])
        negative_img, _ = self._load_single_item(negative_idx)

        return anchor_img, positive_img, negative_img, anchor_label

    def _get_siamese_item(self, idx):
        img0_path, img0_label = os.path.join(self.image_dir, self.image_files[idx]), self.labels[idx]
        should_get_same_class = random.randint(0, 1)

        if should_get_same_class:
            while True:
                img1_idx = random.choice(range(len(self.image_files)))
                if self.labels[img1_idx] == img0_label and img1_idx != idx:
                    break
        else:
            while True:
                img1_idx = random.choice(range(len(self.image_files)))
                if self.labels[img1_idx] != img0_label:
                    break

        img1_path, img1_label = os.path.join(self.image_dir, self.image_files[img1_idx]), self.labels[img1_idx]

        try:
            mode = 'RGB' if self.to_rgb else 'L'
            img0 = Image.open(img0_path).convert(mode)
            img1 = Image.open(img1_path).convert(mode)
        except Exception as e:
            print(f"[Siamese] Error opening {img0_path} or {img1_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.image_files))

        if self.transform:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return (
            img0,
            img1,
            torch.tensor([int(img0_label != img1_label)], dtype=torch.float32),
            img0_label,
            img1_label
        )

    def _load_single_item(self, idx):
        filename = self.image_files[idx]
        label = self.label_map[filename]
        path = os.path.join(self.image_dir, filename)

        try:
            mode = 'RGB' if self.to_rgb else 'L'
            img = Image.open(path).convert(mode)
        except Exception as e:
            print(f"[Triplet] Error opening {filename}: {e}")
            return self._load_single_item((idx + 1) % len(self.image_files))

        if self.transform:
            img = self.transform(img)

        return img, label


def get_partitioned_dataloaders(
    image_dir, label_file, partition_file, transform, batch_size=32, m_per_sample=2,
    num_identities=500, seed=42
):
    """
    Creates train/val/test dataloaders with disjoint identities in each split.
    """
    # Load full dataset (to get all image-label mappings)
    full_dataset = CelebALabeledDataset(
        image_dir, label_file, partition_file=partition_file, partition_id=0, transform=transform
    )

    # Fix seed
    random.seed(seed)
    torch.manual_seed(seed)

    # Get unique labels (identities)
    all_labels = list(set(full_dataset.labels))
    selected_labels = random.sample(all_labels, min(num_identities, len(all_labels)))

    # Split identities into disjoint sets
    num_train = int(0.8 * len(selected_labels))
    num_val = int(0.1 * len(selected_labels))
    num_test = len(selected_labels) - num_train - num_val

    random.shuffle(selected_labels)
    train_ids = set(selected_labels[:num_train])
    val_ids = set(selected_labels[num_train:num_train + num_val])
    test_ids = set(selected_labels[num_train + num_val:])

    def create_split_dataset(split_ids, transform):
        # Get image files and remap labels
        image_files = [f for f, l in zip(full_dataset.image_files, full_dataset.labels) if l in split_ids]
        label_map = {f: full_dataset.label_map[f] for f in image_files}
        original_labels = [label_map[f] for f in image_files]
        unique_labels = sorted(set(original_labels))
        label_remap = {orig: new for new, orig in enumerate(unique_labels)}
        remapped_label_map = {f: label_remap[l] for f, l in label_map.items()}

        class SplitDataset(CelebALabeledDataset):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.image_files = image_files
                self.label_map = remapped_label_map
                self.labels = [self.label_map[f] for f in self.image_files]
                self.unique_labels = torch.unique(torch.tensor(self.labels)).tolist()
                self.label_to_indices = {label: [] for label in self.unique_labels}
                for idx, label in enumerate(self.labels):
                    self.label_to_indices[label].append(idx)

        return SplitDataset(image_dir, label_file, partition_file=partition_file, partition_id=0, transform=transform)

    train_dataset = create_split_dataset(train_ids, train_transform)
    val_dataset = create_split_dataset(val_ids, eval_transform)
    test_dataset = create_split_dataset(test_ids, eval_transform)
    from collections import Counter
    # ---------- FILTER TRAIN SET ----------
    label_counts_train = Counter(train_dataset.labels)
    valid_labels_train = [l for l in label_counts_train if label_counts_train[l] >= m_per_sample]

    filtered_indices_train = [i for i, label in enumerate(train_dataset.labels) if label in valid_labels_train]
    filtered_train_dataset = Subset(train_dataset, filtered_indices_train)

    train_sampler = MPerClassSampler(
        labels=[train_dataset.labels[i] for i in filtered_indices_train],
        m=m_per_sample,
        batch_size=batch_size,
        length_before_new_iter=len(filtered_train_dataset)
    )

    train_loader = DataLoader(filtered_train_dataset, batch_size=batch_size, sampler=train_sampler, drop_last=True)

    # ---------- FILTER VAL SET ----------
    label_counts_val = Counter(val_dataset.labels)
    valid_labels_val = [l for l in label_counts_val if label_counts_val[l] >= m_per_sample]

    filtered_indices_val = [i for i, label in enumerate(val_dataset.labels) if label in valid_labels_val]
    filtered_val_dataset = Subset(val_dataset, filtered_indices_val)

    val_sampler = MPerClassSampler(
        labels=[val_dataset.labels[i] for i in filtered_indices_val],
        m=m_per_sample,
        batch_size=batch_size,
        length_before_new_iter=len(filtered_val_dataset)
    )

    val_loader = DataLoader(filtered_val_dataset, batch_size=batch_size, sampler=val_sampler, drop_last=True)

    # ---------- TEST SET (optional filtering?) ----------
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)

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
'''
import yaml

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("mlflow").setLevel(logging.ERROR)  # oder .CRITICAL

# List of modes to run
MODES = [ "ARCFACE_RESNET_3"]   # "_OWN", "_RESNET" #"ARCFACE_OWN",

with open("config/config.yml", "r") as f:
    config = yaml.safe_load(f)

    # Extract relevant sections
    PRE = config["PREPROCESSING"]


    # Preprocessing config
    IMAGE_DIR = PRE["image_dir"]
    LABEL_FILE = PRE["label_file"]
    PARTITION_FILE = PRE["partition_file"]
    BATCH_SIZE = 32
    IMAGE_SIZE = PRE["image_size"]
    M_PER_SAMPLE = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_IDENTITY = 500     # Number of unique identities in training

    # Set or create experiment

    # Load train, validation, and t est dataloaders
    train_loader, val_loader, _ = get_partitioned_dataloaders(
        image_dir=IMAGE_DIR,
        label_file=LABEL_FILE,
        partition_file=PARTITION_FILE,
        transform=None,
        m_per_sample=M_PER_SAMPLE,
        batch_size=BATCH_SIZE,
        num_identities=NUM_IDENTITY,
        seed=42
    )
    # Get actual image filenames used in train and val loaders
    train_subset = train_loader.dataset
    val_subset = val_loader.dataset

    train_dataset = train_subset.dataset
    train_indices = train_subset.indices

    val_dataset = val_subset.dataset
    val_indices = val_subset.indices

    train_image_files = {train_dataset.image_files[i] for i in train_indices}
    val_image_files = {val_dataset.image_files[i] for i in val_indices}

    common_files = train_image_files.intersection(val_image_files)

    print(f"Train set image count: {len(train_image_files)}")
    print(f"Val set image count: {len(val_image_files)}")
    print(f"Common images in both: {len(common_files)}")

    if common_files:
        print("Common image filenames:", list(common_files)[:10])  # Show sample
    else:
        print("✅ No overlap in images between train and val loaders.")'''


import os
import shutil
from tqdm import tqdm
from PIL import Image

import os
import shutil
import random
from PIL import Image
from tqdm import tqdm

def save_images_by_label(dataset, output_dir, max_label=500, seed=42, split=True, move=True):
    """
    Saves all images from dataset into folders by label and optionally splits them into train/val/test.

    Args:
        dataset: An instance of a dataset with image_files, labels, and image_dir attributes.
        output_dir (str): Where to save all the data.
        max_label (int): Only process labels ≤ max_label.
        seed (int): Random seed for shuffling.
        split (bool): Whether to split folders into train/val/test.
        move (bool): Whether to move instead of copy when splitting.
    """
    # Step 0: If output_dir exists, delete it completely
    if os.path.exists(output_dir):
        print(f"⚠️ Deleting existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Save images grouped by label
    print("Saving images into label folders...")
    for idx in tqdm(range(len(dataset)), desc="Saving images"):
        try:
            filename = dataset.image_files[idx]
            label = dataset.labels[idx]

            if label > max_label:
                continue

            label_folder = os.path.join(output_dir, str(label))
            os.makedirs(label_folder, exist_ok=True)

            src_path = os.path.join(dataset.image_dir, filename)
            dst_path = os.path.join(label_folder, filename)

            # Verify and save image
            img = Image.open(src_path)
            img.save(dst_path)
        except Exception as e:
            print(f"Failed to save image {filename}: {e}")

    if not split:
        return

    # Step 2: Split folders into train/val/test
    print("\nSplitting folders into train/val/test...")

    label_folders = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]
    label_folders = [f for f in label_folders if f.isdigit() and int(f) <= max_label]

    random.seed(seed)
    label_folders.sort()
    random.shuffle(label_folders)

    total = len(label_folders)
    n_train = int(0.8 * total)
    n_val = int(0.15 * total)
    n_test = total - n_train - n_val

    splits = {
        'train': label_folders[:n_train],
        'val': label_folders[n_train:n_train + n_val],
        'test': label_folders[n_train + n_val:]
    }

    for split_name, folders in splits.items():
        for folder in tqdm(folders, desc=f"Moving to {split_name}"):
            src_folder = os.path.join(output_dir, folder)
            dst_folder = os.path.join(output_dir, split_name, folder)
            os.makedirs(os.path.dirname(dst_folder), exist_ok=True)

            if move:
                shutil.move(src_folder, dst_folder)
            else:
                shutil.copytree(src_folder, dst_folder)

    # Step 3: Remove original label folders (only if copied)
    if not move:
        print("\nRemoving original label folders...")
        for folder in label_folders:
            original_path = os.path.join(output_dir, folder)
            if os.path.isdir(original_path):
                shutil.rmtree(original_path)

    print(f"\n✅ Done! Split {total} folders into: {n_train} train, {n_val} val, {n_test} test.")





image_dir= "data/celeba/img_align_celeba"
label_file= "data/celeba/identity_CelebA.txt"
partition_file= "data/celeba/list_eval_partition.csv"
dataset = CelebALabeledDataset(
    image_dir,
    label_file,
    partition_file,
    partition_id=0
)

save_images_by_label(dataset, output_dir="data/celeba/output_images_by_label", max_label=2000)


    