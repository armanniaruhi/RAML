from src.preprocessing.dataLoader_CelebA import get_partitioned_dataloaders, create_subset_loader
from src.preprocessing.dataLoader_vi import SiameseNetworkDataset
from src.ml.resNet18 import SiameseNetwork
from src.ml.loss_utils import ContrastiveLoss, ArcFaceLoss, MultiSimilarityLoss
import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch
import os

import yaml

# Load config from YAML
with open("config/config.yml", "r") as f:
    config = yaml.safe_load(f)

# Extract sections
PRE = config["PREPROCESSING"]
TRAIN = config["TRAINING"]

# Set constants from preprocessing config
IMAGE_DIR = PRE["image_dir"]
LABEL_FILE = PRE["label_file"]
PARTITION_FILE = PRE["partition_file"]
BATCH_SIZE = PRE["batch_size"]
M_PER_SAMPLE = PRE["m_per_sample"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set constants from training config
LR = TRAIN["lr"]
SCHEDULING = TRAIN["scheduling"]
WEIGHT_DECAY = TRAIN["weight_decay"]
NUM_EPOCHS = TRAIN["num_epochs"]
PATIENCE = TRAIN["patience"]
LOSS_TYPE = TRAIN["loss_type"]
DATASET_NAME = "celebA"# "celebA" # ATT

if DATASET_NAME == "celebA":
    # Load datasets
    train_loader, val_loader, test_loader = get_partitioned_dataloaders(
        image_dir=IMAGE_DIR,
        label_file=LABEL_FILE,
        m_per_sample=M_PER_SAMPLE,
        partition_file=PARTITION_FILE,
        batch_size=BATCH_SIZE,
        output_format = "others"
    )

    train_loader = create_subset_loader(train_loader, 5000)
    test_dataloader= create_subset_loader(train_loader, 1000)
else:
    # Load the training dataset
    folder_dataset = datasets.ImageFolder(root="data/v2/data/faces/training/")

    # Resize the images and transform to tensors
    transformation = transforms.Compose([transforms.Resize((100,100)),
                                        transforms.ToTensor()
                                        ])

    # Initialize the network
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                            transform=transformation)
    train_loader = DataLoader(siamese_dataset, shuffle=True, num_workers=0, batch_size=64)

    # Locate the test dataset and load it into the SiameseNetworkDataset
    folder_dataset_test = datasets.ImageFolder(root="data/v2/data/faces/testing/")
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                            transform=transformation)
    test_dataloader = DataLoader(siamese_dataset, num_workers=0, batch_size=1, shuffle=True)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = SiameseNetwork().to(DEVICE)
loss_type = LOSS_TYPE

if loss_type == "contrastive":
    criterion = ContrastiveLoss(margin=20).to(DEVICE)
elif loss_type == "arcface":
    criterion = ArcFaceLoss(num_classes=10000, embedding_size=256).to(DEVICE)
elif loss_type == "multisimilarity":
    import pytorch_metric_learning.losses as losses
    criterion = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5).to(DEVICE)
optimizer = torch.optim.AdamW(net.parameters(), lr=0.0005)


# Showing images
def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
        
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

# Plotting data
def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()


from tqdm import tqdm

def train():
    counter = []
    loss_history = [] 
    iteration_number = 0

    for epoch in range(NUM_EPOCHS):
        cum_loss = 0
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        
        for i, (img0, img1, label, label0, label1) in enumerate(train_loader):
            img0, img1, label, label0, label1 = img0.to(DEVICE), img1.to(DEVICE), label.to(DEVICE), label0.to(DEVICE), label1.to(DEVICE)
            
            optimizer.zero_grad()
            output1, output2 = net(img0, img1)

            if loss_type == "contrastive":
                loss = criterion(output1, output2, label)
            elif loss_type == "multisimilarity":
                embeddings = torch.cat([output1, output2])
                labels = torch.cat([label0, label1])
                loss = criterion(embeddings, labels)
            elif loss_type == "arcface":
                loss = criterion(output1, output2, label0, label1)

            loss.backward()
            optimizer.step()
            
            cum_loss += loss.item()

            # Logging alle 10 Schritte
            if i % 10 == 0:
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss.item())
                avg_loss = cum_loss / (i + 1)
                print(f"[Epoch {epoch + 1}, Batch {i}] Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}")

    # Ordner anlegen und Modell speichern
    os.makedirs("models", exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, "models/siamese_checkpoint.pth")

    show_plot(counter, loss_history)



# 👇 Das ist entscheidend:
if __name__ == "__main__":
    train()
