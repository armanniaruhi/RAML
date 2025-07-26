"""
Refactored and modularized a training script for Siamese Network with cross-validation,
MLflow logging, and optional pretrained loading.
"""
import os
import random
import warnings
import yaml
import logging

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import colorcet as cc 

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

from src.ml.resNet18 import SiameseNetworkCosine, SiameseNetworkContrastive
from src.preprocessing.dataLoader_Siamase import SiameseNetworkDataset
from src.ml.contrastive_loss import ContrastiveLoss


# Initialize colorama
init(autoreset=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s — %(levelname)s — %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    logger.info(Fore.CYAN + f"Setting random seed: {seed}" + Style.RESET_ALL)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str = 'config.yml') -> dict:
    """Load training configuration from YAML file."""
    logger.info(Fore.CYAN + f"Loading config from {path}..." + Style.RESET_ALL)
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    """Return available device (GPU if available)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_transform() -> transforms.Compose:
    """Define image augmentation and preprocessing pipeline."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
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
    """Train and validate for one fold. Returns metrics dict and the best model state."""
    global avg_loss
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


def train_validate(dataset, config, device):
    """Run K-Fold cross-validation and return summary metrics and best model state."""
    k_folds = config.get('k_folds', 5)
    logger.info(Fore.MAGENTA + f"Starting {k_folds}-fold cross-validation" + Style.RESET_ALL)
    kfold = KFold(n_splits=k_folds, shuffle=True)
    all_results = []
    best_overall_f1 = -1
    best_state = None

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        logger.info(Fore.YELLOW + f"\n--- Fold {fold+1}/{k_folds} ---" + Style.RESET_ALL)
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
    logger.info(Fore.MAGENTA + f"\nAverage best F1 across folds: {avg_f1:.4f}" + Style.RESET_ALL)
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
    _, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
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


from sklearn.manifold import TSNE

def get_tsne_embeddings(dataset, model, device):
    """Helper function to extract embeddings for t-SNE visualization."""
    model.eval()
    embeddings = []
    labels = []

    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    with torch.no_grad():
        for i, (img0, img1, _, l0, l1) in enumerate(loader):
            img0 = img0.to(device)
            emb0, _ = model(img0, img0)
            embeddings.append(emb0.squeeze().cpu().numpy())
            labels.append(l0.item())

    return np.array(embeddings), labels


def plot_tsne(embeddings, labels, save_path):
    """Plot t-SNE visualization of embeddings with distinct colors for each label."""
    logger.info(Fore.CYAN + "Preparing t-SNE plot..." + Style.RESET_ALL)
    labels = np.array(labels)
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)

    # Distinct color mapping
    all_colors = cc.glasbey[:n_classes]
    color_dict = {label: all_colors[i] for i, label in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(8, 6))

    for label in unique_labels:
        mask = labels == label
        ax.scatter(embeddings[mask, 0], embeddings[mask, 1], s=10, color=color_dict[label])

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.tick_params(labelleft=True)

    logger.info(Fore.CYAN + f"Saving t-SNE plot to: {save_path}" + Style.RESET_ALL)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()
    logger.info(Fore.GREEN + "t-SNE plot saved and closed." + Style.RESET_ALL)


def plot_tsne_before_after_separately(embeddings_before, labels_before, embeddings_after, labels_after, loss_type):
    """Plot t-SNE for embeddings before and after training, separately."""
    logger.info(Fore.CYAN + "Preparing t-SNE plots for before and after training..." + Style.RESET_ALL)
    all_embeddings = np.concatenate([embeddings_before, embeddings_after])
    if loss_type == "contrastive":
        metric = "euclidean"
        preplexity = max(50, len(all_embeddings) // 2)
    else:
        metric = "cosine"
        preplexity = min(30, len(all_embeddings) // 2)
    tsne = TSNE(n_components=2, perplexity=preplexity,
                n_iter=500, random_state=42, metric=metric)
    all_embeddings_2d = tsne.fit_transform(all_embeddings)

    n_before = len(embeddings_before)
    emb_before_2d = all_embeddings_2d[:n_before]
    emb_after_2d = all_embeddings_2d[n_before:]

    plot_tsne(emb_before_2d, labels_before, f"results/plots/tsne_plots/tsne_before_{loss_type}.png")
    plot_tsne(emb_after_2d, labels_after, f"results/plots/tsne_plots/tsne_after_{loss_type}.png")


# Main function to run the training pipeline
def main():
    # Initialize global config and suppress warnings
    global config
    # Suppress UserWarnings from PyTorch
    warnings.filterwarnings('ignore', category=UserWarning)
    # Suppress Matplotlib warnings
    mlflow.set_experiment("SiameseTraining")
    logger.info(Fore.MAGENTA + "===== Starting Siamese Training Pipeline =====" + Style.RESET_ALL)
    # Load configuration and set up environment
    config = load_config()
    # Set up device and random seed
    device = get_device()
    #  Set random seed for reproducibility
    set_seed(config.get('SEED', 42))
    # Create necessary directories
    init()
    
    logger.info(Fore.BLUE + "Plotting t-SNE BEFORE training..." + Style.RESET_ALL)

    # Prepare dataset and transformations
    dataset = prepare_dataset(root='dataset/train', transform=create_transform())
    # Prepare t-SNE dataset
    tsne_dataset = prepare_dataset(root='dataset/tsne_dataset', transform=create_transform())

    # Build model and loss function
    model_before, _ = build_model_and_loss(config['LOSS_TYPE'], device)
    logger.info(Fore.BLUE + f"Loss type: {config['LOSS_TYPE']}" + Style.RESET_ALL)
    # Get embeddings for t-SNE before training
    emb_before, lbl_before = get_tsne_embeddings(tsne_dataset, model_before, device)
    
    # Set loss type for logging
    loss_type = "ms" if config['LOSS_TYPE'] == 'MultiSimilarity' else config['LOSS_TYPE'].lower()

    if not config['PRETRAINED']:
        # Start training and validation
        logger.info(Fore.YELLOW + "Starting training and validation..." + Style.RESET_ALL)
        mlflow.log_params(config)
        # Train and validate using K-Fold cross-validation
        train_validate(dataset, config, device)
        print(Fore.GREEN + "Plotting t-SNE AFTER training..." + Style.RESET_ALL)
        # Get embeddings for t-SNE after training
        model_after, _ = build_model_and_loss(config['LOSS_TYPE'], device)
        # Load the best model state
        model_after.load_state_dict(torch.load(f"models/models_{loss_type}/best_model_overall.pt", map_location=device))
        # Get embeddings for t-SNE after training
        emb_after, lbl_after = get_tsne_embeddings(tsne_dataset, model_after, device)
    else:
        # Load pretrained model
        logger.info(Fore.YELLOW + "Loading pretrained model..." + Style.RESET_ALL)
        # Load the pretrained model state
        model, _ = build_model_and_loss(config['LOSS_TYPE'], device)
        # Load the best model state
        model.load_state_dict(torch.load(f"models/models_{loss_type}/best_model_overall.pt", map_location=device, weights_only=True))
        # Get embeddings for t-SNE after loading pretrained model
        logger.info(Fore.BLUE + "Plotting t-SNE AFTER loading pretrained model..." + Style.RESET_ALL)
        emb_after, lbl_after = get_tsne_embeddings(tsne_dataset, model, device)

    plot_tsne_before_after_separately(emb_before, lbl_before, emb_after, lbl_after, loss_type)
    logger.info(Fore.MAGENTA + "===== Training Pipeline Completed =====" + Style.RESET_ALL)
    
if __name__ == '__main__':
    main()
