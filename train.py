"""
Refactored and modularized training script for Siamese Network with cross-validation,
MLflow logging, and optional pretrained loading.
"""
import os
import random
import warnings
import yaml

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import KFold
import pytorch_metric_learning.losses as losses
import mlflow
from colorama import Fore, Style, init
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from src.ml.resNet18 import SiameseNetworkCosine, SiameseNetworkContrastive
from src.preprocessing.dataLoader_Siamase import SiameseNetworkDataset
from src.ml.contrastive_loss import ContrastiveLoss


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str = 'config.yml') -> dict:
    """Load training configuration from YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    """Return available device (GPU if available)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_transform() -> transforms.Compose:
    """Define image augmentation and preprocessing pipeline."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.1, 2.3), value='random')
    ])


def prepare_dataset(root: str, transform: transforms.Compose):
    """Load ImageFolder and wrap in SiameseNetworkDataset."""
    folder = datasets.ImageFolder(root=root, transform=transform)
    return SiameseNetworkDataset(imageFolderDataset=folder, transform=transform)


def build_model_and_loss(loss_type: str, device: torch.device):
    """Instantiate network and criterion based on loss type."""
    if loss_type == 'Contrastive':
        model = SiameseNetworkContrastive().to(device)
        criterion = ContrastiveLoss(margin=1).to(device)
    elif loss_type == 'Circle':
        model = SiameseNetworkCosine().to(device)
        criterion = losses.CircleLoss(m=0.25, gamma=256).to(device)
    elif loss_type == 'MultiSimilarity':
        model = SiameseNetworkCosine().to(device)
        criterion = losses.MultiSimilarityLoss().to(device)
    else:
        raise ValueError("LOSS_TYPE must be one of: 'Contrastive', 'Circle', 'MultiSimilarity'")
    return model, criterion


def train_fold(model, criterion, optimizer, train_loader, val_loader,
               device, loss_type, patience):
    """Train and validate for one fold. Returns metrics dict and best model state."""
    best_val_loss = np.inf
    epochs_no_improve = 0
    metrics = {'train_loss': [], 'val_loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
    best_state = None

    for epoch in range(config['NUM_EPOCHS']):
        # Early stopping check
        if config['EARLY_STOP'] and epochs_no_improve >= patience:
            print(Fore.RED + f"Early stopping at epoch {epoch}" + Style.RESET_ALL)
            break

        # Training with progress bar
        model.train()
        cum_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=Fore.CYAN + f"Epoch {epoch+1}/{config['NUM_EPOCHS']} [Train]" + Style.RESET_ALL)
        for i, (imgs0, imgs1, labels, l0, l1) in pbar:
            imgs0, imgs1 = imgs0.to(device), imgs1.to(device)
            labels = labels.to(device)
            l0, l1 = l0.to(device), l1.to(device)

            optimizer.zero_grad()
            out0, out1 = model(imgs0, imgs1)
            if loss_type == 'Contrastive':
                loss = criterion(out0, out1, labels)
            else:
                emb = torch.cat([out0, out1])
                lbl = torch.cat([l0, l1])
                loss = criterion(emb, lbl)
            loss.backward()
            optimizer.step()

            cum_loss += loss.item()
            mlflow.log_metric('batch_train_loss', loss.item())
            avg_loss = cum_loss / (i+1)
            pbar.set_postfix({'Tr_Loss': f"{avg_loss:.4f}",
                              'Best_Val_Loss': f"{best_val_loss:.4f}"})
        pbar.close()
        metrics['train_loss'].append(avg_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        tp = fp = fn = correct = 0
        with torch.no_grad():
            for imgs0, imgs1, labels, l0, l1 in val_loader:
                imgs0, imgs1 = imgs0.to(device), imgs1.to(device)
                labels = labels.to(device)
                l0, l1 = l0.to(device), l1.to(device)

                out0, out1 = model(imgs0, imgs1)
                dist = F.cosine_similarity(out0, out1)
                preds = (dist < 0.5).int()
                correct += (preds == labels).sum().item()
                tp += ((preds == 1) & (labels == 1)).sum().item()
                fp += ((preds == 1) & (labels == 0)).sum().item()
                fn += ((preds == 0) & (labels == 1)).sum().item()

                if loss_type == 'Contrastive':
                    loss = criterion(out0, out1, labels)
                else:
                    emb = torch.cat([out0, out1])
                    lbl = torch.cat([l0, l1])
                    loss = criterion(emb, lbl)
                val_loss += loss.item()

        avg_val = val_loss / len(val_loader)
        acc = correct / len(val_loader.dataset)
        prec = tp / (tp + fp + 1e-10)
        rec = tp / (tp + fn + 1e-10)
        f1 = 2 * prec * rec / (prec + rec + 1e-10)

        metrics['val_loss'].append(avg_val)
        metrics['accuracy'].append(acc)
        metrics['precision'].append(prec)
        metrics['recall'].append(rec)
        metrics['f1'].append(f1)

        # MLflow logging
        mlflow.log_metrics({
            'epoch_train_loss': avg_loss,
            'epoch_val_loss': avg_val,
            'val_accuracy': acc,
            'val_precision': prec,
            'val_recall': rec,
            'val_f1': f1
        }, step=epoch)

        # Checkpoint and early stop logic
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_no_improve = 0
            best_state = model.state_dict()
        else:
            epochs_no_improve += 1

    return metrics, best_state


def cross_validate(dataset, config, device):
    """Run K-Fold cross-validation and return summary metrics and best model state."""
    k_folds = config.get('k_folds', 5)
    kfold = KFold(n_splits=k_folds, shuffle=True)
    all_results = []
    best_overall_f1 = -1
    best_state = None

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n--- Fold {fold+1}/{k_folds} ---")
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=config['BATCHSIZE'], shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=config['BATCHSIZE'], shuffle=False)

        model, criterion = build_model_and_loss(config['LOSS_TYPE'], device)
        optimizer = optim.AdamW(model.parameters(), lr=config['LR'], weight_decay=config['WEIGHT_DECAY'])

        with mlflow.start_run(run_name=f"Fold_{fold+1}", nested=True):
            mlflow.log_param('fold', fold+1)
            metrics, state = train_fold(
                model, criterion, optimizer,
                train_loader, val_loader,
                device, config['LOSS_TYPE'], config['PATIENCE']
            )
            all_results.append(metrics)

            os.makedirs(f"models_{config['LOSS_TYPE']}", exist_ok=True)
            torch.save(state, f"models_{config['LOSS_TYPE']}/best_fold_{fold+1}.pt")
            fold_f1 = max(metrics['f1'])
            if fold_f1 > best_overall_f1:
                best_overall_f1 = fold_f1
                best_state = state
                torch.save(state, f"models_{config['LOSS_TYPE']}/best_overall.pt")

    avg_f1 = np.mean([max(m['f1']) for m in all_results])
    print(f"\nAverage best F1 across folds: {avg_f1:.4f}")
    return all_results, best_state


def plot_samples_with_metrics(dataset, model, device, n_samples=8, n_cols=4):
    """Plot random sample pairs with Euclidean and Cosine metrics."""
    model.eval()
    pairs = []
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for _ in range(n_samples):
        img0, img1, _, l0, l1 = next(iter(loader))
        img0, img1 = img0.to(device), img1.to(device)
        emb0, emb1 = model(img0, img1)
        euclid = F.pairwise_distance(emb0, emb1).item()
        cos_sim = F.cosine_similarity(emb0, emb1).item()
        pairs.append((img0.cpu(), img1.cpu(), l0.item(), l1.item(), euclid, cos_sim))

    n_rows = (n_samples + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axs = axs.flatten() if n_samples>1 else [axs]
    for i, ax in enumerate(axs):
        if i < len(pairs):
            img0, img1, l0, l1, euclid, cos_sim = pairs[i]
            g0 = img0.mean(dim=1).squeeze()
            g1 = img1.mean(dim=1).squeeze()
            cat = torch.cat([g0, g1], dim=1)
            ax.imshow(cat, cmap='gray')
            ax.set_title(f"{l0} vs {l1}\nEuc: {euclid:.2f}, Cos: {cos_sim:.2f}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def main():
    global config
    config = load_config()
    device = get_device()
    set_seed(config.get('SEED', 42))
    warnings.filterwarnings('ignore', category=UserWarning)
    init()  # colorama

    dataset = prepare_dataset(root='dataset/train', transform=create_transform())

    if not config['PRETRAINED']:
        all_results, best_state = cross_validate(dataset, config, device)
        print("Training complete. Best model saved.")
    else:
        print(Fore.YELLOW + "Loading pretrained model..." + Style.RESET_ALL)
        os.makedirs(f"models_{config['LOSS_TYPE']}", exist_ok=True)
        model, _ = build_model_and_loss(config['LOSS_TYPE'], device)
        model.load_state_dict(torch.load(f"models_{config['LOSS_TYPE']}/best_overall.pt", map_location=device))
        plot_samples_with_metrics(dataset, model, device)

if __name__ == '__main__':
    main()
