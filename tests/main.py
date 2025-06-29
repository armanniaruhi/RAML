import torch
import yaml
from src.preprocessing.dataLoader_vi import SiameseNetworkDataset
from src.ml.own_network import SiameseNetworkOwn
from src.ml.resNet18 import SiameseNetwork
from src.ml.loss_utils import ContrastiveLoss, ArcFaceLoss
import pytorch_metric_learning.losses as losses
from tqdm import tqdm
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
import tempfile
import os
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from colorama import Fore, Style, init
init(autoreset=True)  # Automatically reset to default color after each print

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("mlflow").setLevel(logging.ERROR)  # oder .CRITICAL

# List of modes to run
MODES = [ "ARCFACE_RESNET_5"]   # "_OWN", "_RESNET" #"ARCFACE_OWN",

import random
import numpy as np
import torch

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
    #RandomRotate(degrees=10),  # Increased from 2
    #CenterZoom(),
    #transforms.RandomHorizontalFlip(),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    #CenterZoom(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # if you use GPU
    torch.cuda.manual_seed_all(seed)  # if multiple GPUs
    torch.backends.cudnn.deterministic = True  # makes results reproducible but slower
    torch.backends.cudnn.benchmark = False

# Call it once near the top before training or data loading:
set_seed(42)

def run_experiment(MODE):
    # Load configuration parameters from YAML file
    with open("config/config.yml", "r") as f:
        config = yaml.safe_load(f)

        # Extract relevant sections
        PRE = config["PREPROCESSING"]
        if "ARCFACE" in MODE:
            TRAIN = config["TRAINING_ARCFACE"]
            LOSS_TYPE = "arcface"
        elif "CONTRASTIVE" in MODE:
            TRAIN = config["TRAINING_CONTRASTIVE"]
            LOSS_TYPE = "contrastive"
        elif "MS" in MODE:
            TRAIN = config["TRAINING_MS"]
            LOSS_TYPE = "multisimilarity"


        # Preprocessing config
        BATCH_SIZE = 64
        IMAGE_SIZE = PRE["image_size"]
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training config
        NUM_EPOCHS = 20
        PATIENCE = 10
        NUM_IDENTITY = 500     # Number of unique identities in training
        if "RESNET" in MODE:
            NETWORK = "resnet"
        elif "OWN" in MODE:
            NETWORK = "own"

        print(Fore.CYAN + f"\nConfiguring for mode: {MODE}")
        print(Fore.YELLOW + str(TRAIN))

    # Set or create experiment
    experiment_name = MODE
    mlflow.set_experiment(experiment_name)
    # Load the training dataset
    folder_dataset = datasets.ImageFolder(root="data/faces/training")

    # Resize the images and transform to tensors
    transformation = transforms.Compose([transforms.Resize((100,100)),
                                        transforms.ToTensor()
                                        ])

    # Initialize the network
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                            transform=transformation)
    # Load the training dataset
    train_loader = DataLoader(siamese_dataset, shuffle=True, num_workers=0, batch_size=BATCH_SIZE)

    # Locate the test dataset and load it into the SiameseNetworkDataset
    folder_val_dataset = datasets.ImageFolder(root="data/faces/testing/")
    test_dataset = SiameseNetworkDataset(imageFolderDataset=folder_val_dataset,
                                            transform=transformation)
    val_loader = DataLoader(test_dataset, num_workers=0, batch_size=BATCH_SIZE, shuffle=True)


    # Initialize the model
    if NETWORK == "resnet":
        net = SiameseNetwork().to(DEVICE)
    elif NETWORK == "own":
        net = SiameseNetworkOwn().to(DEVICE)

    print(Fore.GREEN + f"Selected Network is: {NETWORK}")

    # Select the loss function
    if LOSS_TYPE == "contrastive":
        criterion = ContrastiveLoss(margin=1).to(DEVICE)
    elif LOSS_TYPE == "arcface":
        criterion = ArcFaceLoss().to(DEVICE)
    elif LOSS_TYPE == "multisimilarity":
        criterion = losses.MultiSimilarityLoss().to(DEVICE)

    # Initialize loss history trackers
    batch_loss_history = []
    batch_val_loss_history = []
    epoch_loss_history = []
    epoch_val_loss_history = []
    # Metric histories (for plotting)
    val_accuracy_history = []
    val_precision_history = []
    val_recall_history = []
    val_f1_history = []


    # Early stopping variables
    best_val_loss = np.inf
    epochs_no_improve = 0
    early_stop = False

    # Optimizer and learning rate scheduler
    # (b) Adjust Learning Rate
    #0.9 and weight decay to 5e−4.
    #For the ArcFace training, we employ the SGD optimizer 
    #optimizer = torch.optim.SGD(net.parameters(), lr=0.0005)  # 3x higher than current
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    from torch.optim.lr_scheduler import MultiStepLR

    # Optimizer (SGD with momentum and weight decay)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)


    # Start MLflow run for tracking
    with mlflow.start_run():
        mlflow.log_param("num_epochs", NUM_EPOCHS)
        mlflow.log_param("loss_type", LOSS_TYPE)
        mlflow.log_param("network", NETWORK)
        mlflow.log_param("optimizer", optimizer.__class__.__name__)
        mlflow.log_param("patience", PATIENCE)

    for epoch in range(NUM_EPOCHS):
        if early_stop:
            print(Fore.RED + f"Early stopping triggered after {epoch} epochs!")
            break

        net.train()
        cum_loss = 0
        pbar = tqdm(enumerate(train_loader, 0), total=len(train_loader),
                    desc=Fore.CYAN + f"{MODE} - Epoch {epoch + 1}/{NUM_EPOCHS} [Training]" + Style.RESET_ALL, leave=True)

        for i, (img0, img1, label, label0, label1) in pbar:
            # Move data to the selected device
            img0, img1, label = img0.to(DEVICE), img1.to(DEVICE), label.to(DEVICE)
            label0, label1 = label0.to(DEVICE), label1.to(DEVICE)

            # Forward pass and compute loss
            optimizer.zero_grad()
            output1, output2 = net(img0, img1)

            if LOSS_TYPE == "contrastive":
                loss= criterion(output1, output2, label)
            else:
                embeddings = torch.cat([output1, output2])
                labels = torch.cat([label0, label1])
                loss = criterion(embeddings, labels)

            # Backpropagation and update
            loss.backward()
            optimizer.step()

            cum_loss += loss.item()
            batch_loss_history.append(loss.item())

            # Log batch loss to MLflow
            mlflow.log_metric("batch_train_loss", loss.item(), step=epoch * len(train_loader) + i)

            avg_train_loss = cum_loss / (i + 1)
            pbar.set_postfix({
                'TrL': f"{avg_train_loss:.4f}",  # Current epoch train loss
                'VL': f"{epoch_val_loss_history[-1]:.4f}" if epoch_val_loss_history else '--',
                'Acc': f"{val_accuracy_history[-1]:.4f}" if val_accuracy_history else '--',
                'P': f"{val_precision_history[-1]:.4f}" if val_precision_history else '--',
                'R': f"{val_recall_history[-1]:.4f}" if val_recall_history else '--',
                'F1': f"{val_f1_history[-1]:.4f}" if val_f1_history else '--',
            })

        epoch_loss_history.append(avg_train_loss)

        # Validation phase
        net.eval()
        val_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        with torch.no_grad():
            for img0, img1, label, label0, label1 in val_loader:
                img0, img1, label = img0.to(DEVICE), img1.to(DEVICE), label.to(DEVICE)
                label0, label1 = label0.to(DEVICE), label1.to(DEVICE)

                output1, output2 = net(img0, img1)
                distances = F.pairwise_distance(output1, output2)  # Fixed distance calculation
                
                # Process each sample in the batch
                for i in range(len(distances)):
                    euclid = distances[i].item()  # Get scalar distance for this pair
                    label_pred = 0 if euclid <= 0.5 else 1
                    true_label = label[i].item()
                
                 # Update metrics
                total_samples += 1
                correct_predictions += (label_pred == true_label)
                
                if label_pred == 1:
                    if true_label == 1:
                        true_positives += 1
                    else:
                        false_positives += 1
                elif true_label == 1:
                    false_negatives += 1

                if LOSS_TYPE == "contrastive":
                    loss = criterion(output1, output2, label)
                else:
                    embeddings = torch.cat([output1, output2])
                    labels = torch.cat([label0, label1])
                    loss = criterion(embeddings, labels)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        epoch_val_loss_history.append(avg_val_loss)
        
        # Calculate metrics
        val_accuracy = correct_predictions / total_samples
        val_precision = true_positives / (true_positives + false_positives + 1e-10)
        val_recall = true_positives / (true_positives + false_negatives + 1e-10)
        val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-10)


        # Append to history lists
        val_accuracy_history.append(val_accuracy)
        val_precision_history.append(val_precision)
        val_recall_history.append(val_recall)
        val_f1_history.append(val_f1)


        # Log all metrics
        mlflow.log_metrics({
            "epoch_train_loss": avg_train_loss,
            "epoch_val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
        }, step=epoch)

        pbar.set_postfix({
            'Train Loss': f"{avg_train_loss:.4f}",
            'Val Loss': f"{avg_val_loss:.4f}",
            'Accuracy': f"{val_accuracy:.4f}",
            'F1': f"{val_f1:.4f}",
            'Patience': f"{epochs_no_improve}/{PATIENCE}"
        })
        pbar.close()

        # Plot metrics
        plt.figure(figsize=(18, 6))
        
        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(epoch_loss_history, label='Train Loss')
        plt.plot(epoch_val_loss_history, label='Val Loss')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Accuracy plot
        plt.subplot(1, 3, 2)
        plt.plot([val_accuracy], label='Accuracy')  # You'll need to store history
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # F1 score plot
        plt.subplot(1, 3, 3)
        plt.plot([val_f1], label='F1 Score')  # You'll need to store history
        plt.title('Validation F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()

        plt.tight_layout()
        
        # Save and log plots
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = os.path.join(tmpdir, f"metrics_plot_{MODE}.png")
            plt.savefig(plot_path)
            mlflow.log_artifact(plot_path, artifact_path="plots")


def main():
    for mode in MODES:
        print(Fore.MAGENTA + f"\n{'='*50}")
        print(Fore.BLUE + f"Starting training for mode: {mode}")
        print(Fore.MAGENTA + f"{'='*50}")
        run_experiment(mode)
        print(Fore.GREEN + f"\nCompleted training for mode: {mode}")

if __name__ == "__main__":
    main()