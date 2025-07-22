# Standard libraries
import random
import warnings
import yaml

# Third-party libraries
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import pytorch_metric_learning.losses as losses
from sklearn.model_selection import KFold
import mlflow
from colorama import Fore, Style, init
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# Local modules
from src.ml.resNet18 import SiameseNetworkCosine, SiameseNetworkContrastive
from src.preprocessing.dataLoader_Siamase import SiameseNetworkDataset
from src.ml.contrastive_loss import ContrastiveLoss

# Set up device and model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the Siamese Network

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

LOSS_TYPE = config["LOSS_TYPE"]
EARLY_STOP = config["EARLY_STOP"]
PATIENCE = config["PATIENCE"]
NUM_EPOCHS = config["NUM_EPOCHS"]
BATCHSIZE = config["BATCHSIZE"]
LR = config["LR"]
WEIGHT_DECAY = config["WEIGHT_DECAY"]
PRETRAINED = config["PRETRAINED"]
    

if LOSS_TYPE not in ["Circle", "MultiSimilarity", "Contrastive"]:
    raise ValueError("Invalid loss type. Choose from 'Circle', 'MultiSimilarity', or 'Contrastive'.")
elif LOSS_TYPE == "Contrastive":
    net = SiameseNetworkContrastive().to(DEVICE)
    criterion = ContrastiveLoss(margin=1).to(DEVICE)
elif LOSS_TYPE == "Circle":
    criterion = losses.CircleLoss(m = 0.25, gamma = 256).to(DEVICE)
    net = SiameseNetworkCosine().to(DEVICE)
else:
    criterion = losses.MultiSimilarityLoss().to(DEVICE)
    net = SiameseNetworkCosine().to(DEVICE)

def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seed before any operations
set_seed(42)

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Transformation pipeline
transformation = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.1, 2.3), value='random')
])

# Prepare the full dataset
full_folder_dataset = datasets.ImageFolder(root="dataset/train", transform=transformation)
full_siamese_dataset = SiameseNetworkDataset(imageFolderDataset=full_folder_dataset,
                                        transform=transformation)
    
# Load pretrained model if specified
if not PRETRAINED:
    # Initialize colorama
    init()

    # Set up K-Fold cross validation
    k_folds = 5
    kfold = KFold(n_splits=k_folds, shuffle=True)

    # MLflow setup
    experiment_name = f"{LOSS_TYPE}_loss"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    mlflow.set_experiment(experiment_name)

    # Training variables
    best_overall_f1 = -1
    best_fold_index = -1
    all_fold_results = []
    radians = []

    for fold, (train_ids, val_ids) in enumerate(kfold.split(full_siamese_dataset)):
        print(f'\n{"="*40}')
        print(f'FOLD {fold + 1}/{k_folds}')
        print(f'{"="*40}\n')
        
        # Initialize metrics for this fold
        batch_loss_history = []
        batch_val_loss_history = []
        epoch_loss_history = []
        epoch_val_loss_history = []
        val_accuracy_history = []
        val_precision_history = []
        val_recall_history = []
        val_f1_history = []
        
        # Create subsets and dataloaders
        train_subsampler = Subset(full_siamese_dataset, train_ids)
        val_subsampler = Subset(full_siamese_dataset, val_ids)
        
        train_dataloader = DataLoader(train_subsampler, batch_size=BATCHSIZE, shuffle=True)
        val_dataloader = DataLoader(val_subsampler, batch_size=BATCHSIZE, shuffle=False)
        
        # Reinitialize model and optimizer for each fold
        if LOSS_TYPE == "Contrastive":
            net = SiameseNetworkContrastive().to(DEVICE)
        elif LOSS_TYPE == "Circle":
            net = SiameseNetworkCosine().to(DEVICE)
        else:
            net = SiameseNetworkCosine().to(DEVICE)
        optimizer = optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        
        
        # Early stopping variables
        best_val_loss = np.inf
        epochs_no_improve = 0
        
        # MLflow tracking for this fold
        with mlflow.start_run(run_name=f"Fold_{fold}", nested=True):
            mlflow.log_param("fold", fold)
            for epoch in range(NUM_EPOCHS):
                if EARLY_STOP:
                    print(Fore.RED + f"Early stopping triggered after {epoch} epochs!")
                    break

                # Training phase
                net.train()
                cum_loss = 0
                pbar = tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader),
                            desc=Fore.CYAN + f"Epoch {epoch + 1}/{NUM_EPOCHS} [Training]" + Style.RESET_ALL, leave=True)

                for i, (img0, img1, label, label0, label1) in pbar:
                    img0, img1, label = img0.to(DEVICE), img1.to(DEVICE), label.to(DEVICE)
                    label0, label1 = label0.to(DEVICE), label1.to(DEVICE)

                    optimizer.zero_grad()
                    output1, output2 = net(img0, img1)

                    if LOSS_TYPE == "Contrastive":
                        loss = criterion(output1, output2, label)
                    else:
                        embeddings = torch.cat([output1, output2])
                        labels = torch.cat([label0, label1])
                        loss = criterion(embeddings, labels)

                    loss.backward()
                    optimizer.step()

                    cum_loss += loss.item()
                    batch_loss_history.append(loss.item())
                    mlflow.log_metric("batch_train_loss", loss.item(), step=epoch * len(train_dataloader) + i)

                    avg_train_loss = cum_loss / (i + 1)
                    pbar.set_postfix({
                        'TrL': f"{avg_train_loss:.4f}",
                        'VL': f"{epoch_val_loss_history[-1]:.4f}" if epoch_val_loss_history else '--',
                        'Acc': f"{val_accuracy_history[-1]:.4f}" if val_accuracy_history else '--',
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
                
                # For ROC AUC calculation
                all_distances = []
                all_labels = []

                with torch.no_grad():
                    for img0, img1, label, label0, label1 in val_dataloader:
                        img0, img1, label = img0.to(DEVICE), img1.to(DEVICE), label.to(DEVICE)
                        label0, label1 = label0.to(DEVICE), label1.to(DEVICE)

                        output1, output2 = net(img0, img1)
                        distances =F.cosine_similarity(output1, output2)
                
                        
                        # Store distances and labels for ROC AUC
                        all_distances.extend(distances.cpu().numpy())
                        all_labels.extend(label.cpu().numpy())
                        
                        for i in range(len(distances)):
                            euclid = distances[i].item()
                            label_pred = 0 if euclid >= 0.5 else 1
                            true_label = label[i].item()
                        
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

                avg_val_loss = val_loss / len(val_dataloader)
                epoch_val_loss_history.append(avg_val_loss)
                
                # Calculate metrics
                val_accuracy = correct_predictions / total_samples
                val_precision = true_positives / (true_positives + false_positives + 1e-10)
                val_recall = true_positives / (true_positives + false_negatives + 1e-10)
                val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-10)
                
        
                
                # Update histories
                val_accuracy_history.append(val_accuracy)
                val_precision_history.append(val_precision)
                val_recall_history.append(val_recall)
                val_f1_history.append(val_f1)

                # Log metrics to MLflow
                mlflow.log_metrics({
                    "epoch_train_loss": avg_train_loss,
                    "epoch_val_loss": avg_val_loss,
                    "val_accuracy": val_accuracy,
                    "val_precision": val_precision,
                    "val_recall": val_recall,
                    "val_f1": val_f1,
                }, step=epoch)

                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    torch.save(net.state_dict(), f"models_{LOSS_TYPE}/best_model_fold_{fold}.pt")
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= PATIENCE:
                        EARLY_STOP = True

                pbar.set_postfix({
                    'Train Loss': f"{avg_train_loss:.4f}",
                    'Val Loss': f"{avg_val_loss:.4f}",
                    'Accuracy': f"{val_accuracy:.4f}",
                    'F1': f"{val_f1:.4f}",
                })
                pbar.close()

            # Store fold results
            fold_results = {
                'best_val_loss': best_val_loss,
                'best_val_f1': max(val_f1_history),
                'best_val_accuracy': max(val_accuracy_history),
                'final_epoch': epoch,
                'loss_history': epoch_loss_history,
                'val_loss_history': epoch_val_loss_history,
                'metrics_history': {
                    'accuracy': val_accuracy_history,
                    'f1': val_f1_history,
                    'precision': val_precision_history,
                    'recall': val_recall_history,
                }
            }
            all_fold_results.append(fold_results)
            
            # Track best overall model
            current_fold_best_f1 = max(val_f1_history)
            if current_fold_best_f1 > best_overall_f1:
                best_overall_f1 = current_fold_best_f1
                best_fold_index = fold
                torch.save(net.state_dict(), f"models_{LOSS_TYPE}/best_model_overall.pt")
                mlflow.log_artifact(f"models_{LOSS_TYPE}/best_model_overall.pt")

    # After all folds complete
    print("\nCross-Validation Results Summary:")
    print("="*50)
    for i, res in enumerate(all_fold_results):
        print(f"Fold {i}: Best Val Loss: {res['best_val_loss']:.4f}, "
            f"Best Val F1: {res['best_val_f1']:.4f}, "
            f"Best Accuracy: {res['best_val_accuracy']:.4f}")

    # Calculate and log average metrics
    avg_val_loss = np.mean([res['best_val_loss'] for res in all_fold_results])
    avg_val_f1 = np.mean([res['best_val_f1'] for res in all_fold_results])
    avg_val_acc = np.mean([res['best_val_accuracy'] for res in all_fold_results])

    print(f"\nAverage across all folds: Val Loss: {avg_val_loss:.4f}, "
        f"Val F1: {avg_val_f1:.4f}, Accuracy: {avg_val_acc:.4f}")

    # Log the average metrics to MLflow
    with mlflow.start_run(run_name="CV_Summary"):
        mlflow.log_metrics({
            "avg_val_loss": avg_val_loss,
            "avg_val_f1": avg_val_f1,
            "avg_val_accuracy": avg_val_acc,
        })
        mlflow.log_params({
            "k_folds": k_folds,
            "optimizer": "AdamW",
            "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY
        })

    print(f"\nBest model from Fold {best_fold_index} (Val F1={best_overall_f1:.4f})")

else:
    print(Fore.YELLOW + "Loading pretrained model weights..." + Style.RESET_ALL)
    if LOSS_TYPE == "CONTRASTIVE":
        model_path = "models/models_contrastive/best_model_overall.pt"
        net = SiameseNetworkContrastive().to(DEVICE)

    elif LOSS_TYPE == "MultiSimilarity":
        model_path = "models/models_ms/models_ms_40_80/best_model_overall.pt"
        net = SiameseNetworkCosine().to(DEVICE)
    else:
        model_path = "models/models_circle/best_model_overall.pt"
        net = SiameseNetworkCosine().to(DEVICE)
    net.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)   

# Set model to evaluation mode
net.eval()

# Collect embeddings and labels from validation set
all_embeddings = []
all_labels = []
with torch.no_grad():
    for img0, img1, _, label0, label1 in full_siamese_dataset:
        img0 = img0.unsqueeze(0).to(DEVICE)
        img1 = img1.unsqueeze(0).to(DEVICE)

        # Convert int to tensors before using .to(DEVICE)
        label0 = torch.tensor([label0], device=DEVICE)
        label1 = torch.tensor([label1], device=DEVICE)

        output1, output2 = net(img0, img1)

        all_embeddings.append(torch.cat([output1, output2], dim=1))
        all_labels.append(torch.cat([label0, label1]))
        print(f"Processed images with labels: {label0.item()} and {label1.item()}")


# Concatenate all embeddings and labels
all_embeddings = torch.cat(all_embeddings)  # shape: [N, D]
all_labels = torch.cat(all_labels)          # shape: [N]


def process_label(lbl):
    if isinstance(lbl, torch.Tensor):
        return lbl[0].cpu().item() if lbl.dim() > 0 else lbl.item()
    return lbl

def get_embeddings(model, img1, img2):
    """Get embeddings from the model for both images"""
    with torch.no_grad():
        model.eval()
        # Assuming your model takes one image at a time and returns its embedding
        emb1, emb2 = model(img1, img2)
    return emb1, emb2

def plot_samples_with_metrics(test_loader, model, n_samples=8, n_cols=4):
    pairs_imgs = []
    all_metrics = []

    for _ in range(n_samples):
        img1, img2, _, label1, label2 = next(iter(test_loader))
        img1 = img1.unsqueeze(0).to(DEVICE)
        img2 = img2.unsqueeze(0).to(DEVICE)

        # Convert int to tensors before using .to(DEVICE)
        label1 = torch.tensor([label1], device=DEVICE)
        label2 = torch.tensor([label2], device=DEVICE)
        lbl1, lbl2 = process_label(label1), process_label(label2)

        # Get embeddings and calculate metrics
        emb1, emb2 = get_embeddings(model, img1, img2)
        euclid = F.pairwise_distance(emb1, emb2).mean().item()
        cos_sim = F.cosine_similarity(emb1, emb2).mean().item()

        # Convert images to grayscale and concatenate
        img1_gray = img1[0].cpu().mean(dim=0, keepdim=True)
        img2_gray = img2[0].cpu().mean(dim=0, keepdim=True)
        concat_img = torch.cat((img1_gray, img2_gray), dim=2)
        pairs_imgs.append((concat_img.squeeze(0), lbl1, lbl2, euclid, cos_sim))

    # Create figure
    n_rows = (n_samples + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 5 * n_rows))
    if n_samples > 1:
        axs = axs.flatten()
    else:
        axs = [axs]

    for i, ax in enumerate(axs):
        if i < n_samples:
            img, lbl1, lbl2, euclid, cos_sim = pairs_imgs[i]
            
            # Display image
            ax.imshow(img.numpy(), cmap='gray')
            ax.axis('off')
            ax.set_title(f"Labels: {lbl1} & {lbl2}", fontsize=10)

            # Display metrics
            text_y = -0.15
            euclid_symbol = "✓" if euclid < 0.5 else "✗"
            euclid_color = "green" if euclid < 0.5 else "red"
            
            cos_sim_symbol = "✓" if cos_sim > 0.45 else "✗"
            cos_sim_color = "green" if cos_sim > 0.45 else "red"
            
            ax.text(0, text_y, f"Euclidean: {euclid:.3f} {euclid_symbol}", 
                   transform=ax.transAxes, fontsize=8, color=euclid_color)
            ax.text(0, text_y - 0.12, f"Cosine: {cos_sim:.3f} {cos_sim_symbol}", 
                   transform=ax.transAxes, fontsize=8, color=cos_sim_color)
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage with your single model
plot_samples_with_metrics(test_loader=full_siamese_dataset, model=net, n_samples=8, n_cols=4)