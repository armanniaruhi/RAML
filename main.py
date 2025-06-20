import torch
import yaml
import os
import numpy as np
import warnings
import logging
import tempfile
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from tqdm import tqdm
from colorama import Fore, Style, init
import optuna

from src.preprocessing.dataLoader_CelebA import get_partitioned_dataloaders
from src.ml.own_network import SiameseNetworkOwn
from src.ml.resNet18 import SiameseNetwork
from src.ml.loss_utils import ContrastiveLoss, ArcFaceLoss
import pytorch_metric_learning.losses as losses

init(autoreset=True)
warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)

MODES = ["ARCFACE_OWN"]

def get_best_val_loss_from_mlflow_or_checkpoint(mode="ARCFACE_OWN", checkpoint_dir="models"):
    checkpoint_path = os.path.join(checkpoint_dir, f"{mode}.pth")
    if not os.path.exists(checkpoint_path):
        print(f"[Warning] Checkpoint not found at {checkpoint_path}")
        return np.inf

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    val_loss_list = checkpoint.get("val_loss", None)

    if val_loss_list is None or not isinstance(val_loss_list, list) or len(val_loss_list) == 0:
        print(f"[Warning] No validation loss history found in checkpoint.")
        return np.inf

    best_val_loss = min(val_loss_list)
    print(f"[Info] Loaded best validation loss from checkpoint: {best_val_loss:.4f}")
    return best_val_loss

def run_experiment(MODE):
    with open("config/config_optuna.yml", "r") as f:
        config = yaml.safe_load(f)

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

    IMAGE_DIR = PRE["image_dir"]
    LABEL_FILE = PRE["label_file"]
    PARTITION_FILE = PRE["partition_file"]
    BATCH_SIZE = PRE["batch_size"]
    IMAGE_SIZE = PRE["image_size"]
    M_PER_SAMPLE = PRE["m_per_sample"]
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LR = TRAIN["lr"]
    SCHEDULING = TRAIN["scheduling"]
    WEIGHT_DECAY = TRAIN["weight_decay"]
    NUM_EPOCHS = TRAIN["num_epochs"]
    PATIENCE = TRAIN["patience"]
    MARGIN = TRAIN.get("margin", 0.5)
    SCALE = TRAIN.get("scale", 64)
    EMBEDDING_SIZE = TRAIN.get("embedding_size", 256)
    NUM_IDENTITY = 200

    NETWORK = "resnet" if "RESNET" in MODE else "own"
    print(Fore.CYAN + f"\nConfiguring for mode: {MODE}")
    print(Fore.YELLOW + str(TRAIN))

    mlflow.set_experiment(MODE)

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

    net = SiameseNetworkOwn().to(DEVICE) if NETWORK == "own" else SiameseNetwork().to(DEVICE)
    print(Fore.GREEN + f"Selected Network is: {NETWORK}")

    if LOSS_TYPE == "contrastive":
        criterion = ContrastiveLoss(margin=20).to(DEVICE)
    elif LOSS_TYPE == "arcface":
        criterion = ArcFaceLoss(num_classes=NUM_IDENTITY, embedding_size=EMBEDDING_SIZE, margin=MARGIN, scale=SCALE).to(DEVICE)
    elif LOSS_TYPE == "multisimilarity":
        criterion = losses.MultiSimilarityLoss(alpha=2, beta=50.0, base=0.5).to(DEVICE)

    batch_loss_history, epoch_loss_history = [], []
    batch_val_loss_history, epoch_val_loss_history = [], []
    best_val_loss, epochs_no_improve, early_stop = np.inf, 0, False

    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    with mlflow.start_run():
        mlflow.log_params({
            "num_epochs": NUM_EPOCHS,
            "loss_type": LOSS_TYPE,
            "network": NETWORK,
            "optimizer": optimizer.__class__.__name__,
            "patience": PATIENCE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "margin": MARGIN,
            "scale": SCALE
        })

        for epoch in range(NUM_EPOCHS):
            if early_stop:
                print(Fore.RED + f"Early stopping triggered after {epoch} epochs!")
                break

            net.train()
            cum_loss = 0
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"{MODE} - Epoch {epoch+1}/{NUM_EPOCHS}")

            for i, (img0, img1, label, label0, label1) in pbar:
                img0, img1, label = img0.to(DEVICE), img1.to(DEVICE), label.to(DEVICE)
                label0, label1 = label0.to(DEVICE), label1.to(DEVICE)

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

                loss.backward()
                optimizer.step()

                cum_loss += loss.item()
                batch_loss_history.append(loss.item())
                mlflow.log_metric("batch_train_loss", loss.item(), step=epoch * len(train_loader) + i)

            avg_train_loss = cum_loss / len(train_loader)
            epoch_loss_history.append(avg_train_loss)

            net.eval()
            val_loss = 0
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

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == PATIENCE:
                    early_stop = True

            mlflow.log_metrics({
                "epoch_train_loss": avg_train_loss,
                "epoch_val_loss": avg_val_loss,
                "best_val_loss": best_val_loss
            }, step=epoch)

            if SCHEDULING:
                scheduler.step()

            checkpoint_path = f"models/{MODE}.pth"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": epoch_loss_history,
                "val_loss": epoch_val_loss_history,
                "batch_train_loss": batch_loss_history
            }, checkpoint_path)

            mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")

            input_example = {
                "input1": torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE).cpu().numpy(),
                "input2": torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE).cpu().numpy()
            }
            mlflow.pytorch.log_model(net, "models/final_model", input_example=input_example)

def objective(trial):
    margin = trial.suggest_float("margin", 0.2, 0.6)
    scale = trial.suggest_float("scale", 30, 64)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    scheduling = trial.suggest_categorical("scheduling", [True, False])
    patience = trial.suggest_int("patience", 7, 25)

    config = {
        "PREPROCESSING": {
            "image_dir": "data/celeba/img_align_celeba",
            "label_file": "data/celeba/identity_CelebA.txt",
            "partition_file": "data/celeba/list_eval_partition.csv",
            "batch_size": 64,
            "m_per_sample": 2,
            "image_size": 100
        },
        "TRAINING_ARCFACE": {
            "lr": lr,
            "weight_decay": weight_decay,
            "scheduling": scheduling,
            "num_epochs": 100,
            "patience": patience,
            "margin": margin,
            "scale": scale,
            "embedding_size": 256
        }
    }

    with open("config/config_optuna.yml", "w") as f:
        yaml.dump(config, f)

    run_experiment("ARCFACE_OWN")
    return get_best_val_loss_from_mlflow_or_checkpoint("ARCFACE_OWN")

def main():
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=25)
    print("Best hyperparameters:", study.best_params)
    print("Best validation loss:", study.best_value)

if __name__ == "__main__":
    main()