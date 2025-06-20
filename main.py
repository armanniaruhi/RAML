import torch
import yaml
from src.preprocessing.dataLoader_CelebA import get_partitioned_dataloaders
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
from colorama import Fore, Style, init
init(autoreset=True)  # Automatically reset to default color after each print

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("mlflow").setLevel(logging.ERROR)  # oder .CRITICAL

# List of modes to run
MODES = ["ARCFACE_OWN"]   # "_OWN", "_RESNET" #"ARCFACE_OWN",

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
        IMAGE_DIR = PRE["image_dir"]
        LABEL_FILE = PRE["label_file"]
        PARTITION_FILE = PRE["partition_file"]
        BATCH_SIZE = PRE["batch_size"]
        IMAGE_SIZE = PRE["image_size"]
        M_PER_SAMPLE = PRE["m_per_sample"]
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Training config
        LR = TRAIN["lr"]
        SCHEDULING = TRAIN["scheduling"]
        WEIGHT_DECAY = TRAIN["weight_decay"]
        NUM_EPOCHS = TRAIN["num_epochs"]
        PATIENCE = TRAIN["patience"]
        NUM_IDENTITY = 200     # Number of unique identities in training
        if "RESNET" in MODE:
            NETWORK = "resnet"
        elif "OWN" in MODE:
            NETWORK = "own"

        print(Fore.CYAN + f"\nConfiguring for mode: {MODE}")
        print(Fore.YELLOW + str(TRAIN))
        
    # Set or create experiment
    experiment_name = MODE
    mlflow.set_experiment(experiment_name)

    # Load train, validation, and test dataloaders
    train_loader, val_loader, _ = get_partitioned_dataloaders(
        image_dir=IMAGE_DIR,
        label_file=LABEL_FILE,
        partition_file=PARTITION_FILE,
        image_size=IMAGE_SIZE,
        m_per_sample=M_PER_SAMPLE,
        batch_size=BATCH_SIZE,
        num_identities=NUM_IDENTITY,
        seed=42
    )

    # Initialize the model
    if NETWORK == "resnet":
        net = SiameseNetwork().to(DEVICE)
    elif NETWORK == "own":
        net = SiameseNetworkOwn().to(DEVICE)
    
    print(Fore.GREEN + f"Selected Network is: {NETWORK}")

    # Select the loss function
    if LOSS_TYPE == "contrastive":
        criterion = ContrastiveLoss(margin=20).to(DEVICE)
    elif LOSS_TYPE == "arcface":
        criterion = ArcFaceLoss(num_classes=NUM_IDENTITY, embedding_size=256, margin=0.5, scale=64).to(DEVICE)
    elif LOSS_TYPE == "multisimilarity":
        criterion = losses.MultiSimilarityLoss(alpha=2, beta=50.0, base=0.5).to(DEVICE)

    # Initialize loss history trackers
    batch_loss_history = []
    batch_val_loss_history = []
    epoch_loss_history = []
    epoch_val_loss_history = []
    
    # Early stopping variables
    best_val_loss = np.inf
    epochs_no_improve = 0
    early_stop = False

    # Optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

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
                        loss = criterion(output1, output2, label)
                    elif LOSS_TYPE == "multisimilarity":
                        embeddings = torch.cat([output1, output2])
                        labels = torch.cat([label0, label1])
                        loss = criterion(embeddings, labels)
                    elif LOSS_TYPE == "arcface":
                        loss = criterion(output1, output2, label0, label1)

                    # Backpropagation and update
                    loss.backward()
                    optimizer.step()

                    cum_loss += loss.item()
                    batch_loss_history.append(loss.item())

                    # Log batch loss to MLflow
                    mlflow.log_metric("batch_train_loss", loss.item(), step=epoch * len(train_loader) + i)

                    avg_train_loss = cum_loss / (i + 1)
                    pbar.set_postfix({
                        'Train Loss': f"{avg_train_loss:.4f}",
                        'Val Loss': f"{epoch_val_loss_history[-1]:.4f}" if epoch_val_loss_history else '--'
                    })

                epoch_loss_history.append(avg_train_loss)

                # Validation phase
                net.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for img0, img1, label, label0, label1 in val_loader:
                        img0, img1, label = img0.to(DEVICE), img1.to(DEVICE), label.to(DEVICE)
                        label0, label1 = label0.to(DEVICE), label1.to(DEVICE)

                        output1, output2 = net(img0, img1)

                        if LOSS_TYPE == "contrastive":
                            loss = criterion(output1, output2, label)
                        elif LOSS_TYPE == "multisimilarity":
                            embeddings = torch.cat([output1, output2])
                            labels = torch.cat([label0, label1])
                            loss = criterion(embeddings, labels)
                        elif LOSS_TYPE == "arcface":
                            loss = criterion(output1, output2, label0, label1)

                        val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)
                epoch_val_loss_history.append(avg_val_loss)

                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    # Save best model
                    #torch.save(net.state_dict(), f'models/{MODE}.pth')
                    #mlflow.log_artifact(f'models/{MODE}.pth', artifact_path="best_model")
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve == PATIENCE:
                        early_stop = True

                # Log epoch losses
                mlflow.log_metric("epoch_train_loss", avg_train_loss, step=epoch)
                mlflow.log_metric("epoch_val_loss", avg_val_loss, step=epoch)
                mlflow.log_metric("best_val_loss", best_val_loss, step=epoch)

                pbar.set_postfix({
                    'Train Loss': f"{avg_train_loss:.4f}",
                    'Val Loss': f"{avg_val_loss:.4f}",
                    'Best Val Loss': f"{best_val_loss:.4f}",
                    'Patience': f"{epochs_no_improve}/{PATIENCE}"
                })
                pbar.close()

                # Update learning rate scheduler
                if SCHEDULING:
                    scheduler.step()

                # Plot batch and epoch loss curves
                plt.figure(figsize=(12, 6))

                if batch_loss_history:
                    plt.subplot(1, 2, 1)
                    plt.plot(batch_loss_history, label='Batch Train Loss')
                    if batch_val_loss_history:
                        plt.plot(batch_val_loss_history, label='Batch Val Loss')
                    plt.title(f'Batch Loss History - {MODE}')
                    plt.xlabel('Batch Number')
                    plt.ylabel('Loss')
                    plt.legend()

                plt.subplot(1, 2, 2)
                plt.plot(epoch_loss_history, label='Epoch Train Loss')
                plt.plot(epoch_val_loss_history, label='Epoch Val Loss')
                plt.title(f'Epoch Loss History - {MODE}')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()

                plt.tight_layout()

                # Save plots to temporary file and log to MLflow
                with tempfile.TemporaryDirectory() as tmpdir:
                    plot_path = os.path.join(tmpdir, f"loss_plot_{MODE}.png")
                    plt.savefig(plot_path)
                    mlflow.log_artifact(plot_path, artifact_path="plots")

                # Save and log model checkpoint
                checkpoint_path = f'models/{MODE}.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': epoch_loss_history,
                    'val_loss': epoch_val_loss_history,
                    'batch_train_loss': batch_loss_history,
                    'batch_val_loss': batch_val_loss_history if 'batch_val_loss_history' in locals() else None,
                }, checkpoint_path)

                mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")

                # Log final model to MLflow
                # Create input example for logging (dummy images with correct shape)
                # Adjust size if your network expects something different
                input_example = {
                    "input1": torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE).cpu().numpy(), # img0
                    "input2": torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE).cpu().numpy()  # img1
                }

                # Log model with input example so MLflow can infer the signature
                mlflow.pytorch.log_model(net, "models/final_model", input_example=input_example)


def main():
    for mode in MODES:
        print(Fore.MAGENTA + f"\n{'='*50}")
        print(Fore.BLUE + f"Starting training for mode: {mode}")
        print(Fore.MAGENTA + f"{'='*50}")
        run_experiment(mode)
        print(Fore.GREEN + f"\nCompleted training for mode: {mode}")

if __name__ == "__main__":
    main()