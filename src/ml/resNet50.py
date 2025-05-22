import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
from torch.optim import Adam
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_curve, auc, roc_auc_score)
import mlflow
import mlflow.pytorch
from typing import Dict, Optional


class SiameseResNet(nn.Module):
    def __init__(self):
        """
        Description:
        Creates a Siamese network using ResNet50 as the base model.
        1. Input Dimension:
                - The ResNet50 expects input images of size `(batch_size, 3, 224, 224)`
                - 3 channels (RGB)
                - 224x224 pixels (standard ResNet input size)
    
            2. After ResNet backbone (before the fc layers):
                - Shape: `(batch_size, 2048, 1, 1)`
                - The 2048 comes from the final convolutional layer of ResNet50
                - The 1x1 spatial dimensions are the result of the average pooling
    
            3. After flattening:
                - Shape: `(batch_size, 2048)`
                - This flattened vector is fed into the fully connected layers
            
            4. Fully Connected Layers:
                - First FC layer: 2048 → 512
                - ReLU activation maintains 512 dimensions
                - Final FC layer: 512 → 128
                - Final output shape: `(batch_size, 128)`
    
        Args:
            embedding_dim (int): Final embedding dimension (default: 128)
            hidden_dim (int): Hidden layer dimension (default: 512)
        
        Returns:
            model: Configured SiameseResNet model
            criterion: Contrastive loss function

        """
        super(SiameseResNet, self).__init__()
        # Load pretrained ResNet50 with the latest weights
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Remove the final fully connected layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Add a fully connected layer for the final embedding
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        
    def forward_one(self, x):
        # Forward pass for one input
        x = self.resnet(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x
    
    def forward(self, input1, input2):
        # Forward pass for both inputs
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2

    def compute_metrics(self, outputs1: torch.Tensor,
                        outputs2: torch.Tensor,
                        labels: torch.Tensor,
                        threshold: float = 0.5) -> Dict[str, float]:
        """Compute classification metrics including ROC/AUC"""
        cos = nn.CosineSimilarity(dim=1)
        similarities = cos(outputs1, outputs2)

        # Convert to numpy
        similarities_np = similarities.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        # Binary predictions
        preds = (similarities_np > threshold).astype(float)

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(labels_np, preds),
            'precision': precision_score(labels_np, preds, zero_division=0),
            'recall': recall_score(labels_np, preds, zero_division=0),
            'f1': f1_score(labels_np, preds, zero_division=0),
            'roc_auc': roc_auc_score(labels_np, similarities_np)
        }

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(labels_np, similarities_np)
        metrics['roc_curve'] = (fpr, tpr)

        return metrics

    def train_model_constructive(self,
                                 train_loader: torch.utils.data.DataLoader,
                                 val_loader: Optional[torch.utils.data.DataLoader] = None,
                                 criterion: Optional[nn.Module] = None,
                                 learning_rate: float = 0.001,
                                 num_epochs: int = 10,
                                 device: str = 'cuda',
                                 experiment_name: str = 'SiameseResNet',
                                 training_mode: bool = True) -> Optional[Dict]:
        """
        Train or evaluate the model with optional tracking

        Args:
            training_mode: If True, stores metrics, embeddings and logs to MLflow.
                          If False, only computes metrics without storing.
        """
        if training_mode:
            mlflow.set_experiment(experiment_name)
            mlflow.start_run()
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("num_epochs", num_epochs)
            mlflow.log_param("hidden_dim", self.fc[0].out_features)
            mlflow.log_param("embedding_dim", self.fc[2].out_features)

        self.to(device)
        if criterion is None:
            criterion = nn.CosineEmbeddingLoss()
            if training_mode:
                mlflow.log_param("loss_function", "CosineEmbeddingLoss")

        optimizer = Adam(self.parameters(), lr=learning_rate)

        # Initialize results storage only if in training mode
        results = None
        if training_mode:
            results = {
                'metrics': {
                    'train': {'loss': [], 'accuracy': [], 'precision': [],
                              'recall': [], 'f1': [], 'roc_auc': []},
                    'val': {'loss': [], 'accuracy': [], 'precision': [],
                            'recall': [], 'f1': [], 'roc_auc': []}
                },
                'embeddings': {
                    'train': {'embeddings1': [], 'embeddings2': [], 'labels': []},
                    'val': {'embeddings1': [], 'embeddings2': [], 'labels': []}
                },
                'roc_curves': {
                    'train': {'fpr': [], 'tpr': []},
                    'val': {'fpr': [], 'tpr': []}
                }
            }

        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            # Training phase
            self.train()
            train_loss = 0.0
            epoch_train_embeddings1, epoch_train_embeddings2, epoch_train_labels = [], [], []

            pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')

            for img1, img2, label in pbar:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)

                optimizer.zero_grad()
                output1, output2 = self(img1, img2)
                loss = criterion(output1, output2, label)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                if training_mode:
                    epoch_train_embeddings1.append(output1.detach().cpu())
                    epoch_train_embeddings2.append(output2.detach().cpu())
                    epoch_train_labels.append(label.detach().cpu())

                pbar.set_postfix({'loss': train_loss / len(pbar)})

            # Compute training metrics
            train_embeddings1 = torch.cat(epoch_train_embeddings1).to(device) if training_mode else None
            train_embeddings2 = torch.cat(epoch_train_embeddings2).to(device) if training_mode else None
            train_labels = torch.cat(epoch_train_labels).to(device) if training_mode else None

            # For validation, we need to compute metrics even if not storing
            if training_mode or val_loader is not None:
                train_metrics = self.compute_metrics(
                    train_embeddings1 if training_mode else output1,
                    train_embeddings2 if training_mode else output2,
                    train_labels if training_mode else label
                )
                epoch_train_loss = train_loss / len(train_loader)

            if training_mode:
                # Store training results
                results['metrics']['train']['loss'].append(epoch_train_loss)
                for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                    results['metrics']['train'][metric].append(train_metrics[metric])

                results['embeddings']['train']['embeddings1'].extend(epoch_train_embeddings1)
                results['embeddings']['train']['embeddings2'].extend(epoch_train_embeddings2)
                results['embeddings']['train']['labels'].extend(epoch_train_labels)

                results['roc_curves']['train']['fpr'], results['roc_curves']['train']['tpr'] = train_metrics[
                    'roc_curve']

                # Log training metrics to MLflow
                mlflow.log_metric("train_loss", epoch_train_loss, step=epoch)
                for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                    mlflow.log_metric(f"train_{metric}", train_metrics[metric], step=epoch)

            print(f"Train - Loss: {epoch_train_loss:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}, "
                  f"AUC: {train_metrics['roc_auc']:.4f}")

            # Validation phase
            if val_loader is not None:
                val_results = self.evaluate(val_loader, criterion, device, training_mode)
                val_loss = val_results['metrics']['loss']

                if training_mode:
                    # Store validation results
                    results['metrics']['val']['loss'].append(val_loss)
                    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                        results['metrics']['val'][metric].append(val_results['metrics'][metric])

                    results['embeddings']['val']['embeddings1'].extend(val_results['embeddings']['embeddings1'])
                    results['embeddings']['val']['embeddings2'].extend(val_results['embeddings']['embeddings2'])
                    results['embeddings']['val']['labels'].extend(val_results['embeddings']['labels'])

                    results['roc_curves']['val']['fpr'], results['roc_curves']['val']['tpr'] = val_results['metrics'][
                        'roc_curve']

                    # Log validation metrics to MLflow
                    mlflow.log_metric("val_loss", val_loss, step=epoch)
                    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                        mlflow.log_metric(f"val_{metric}", val_results['metrics'][metric], step=epoch)

                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        mlflow.pytorch.log_model(self, "best_model")
                        torch.save(self.state_dict(), 'best_model.pth')

                print(f"Val - Loss: {val_loss:.4f}, "
                      f"Acc: {val_results['metrics']['accuracy']:.4f}, "
                      f"AUC: {val_results['metrics']['roc_auc']:.4f}")

        if training_mode:
            # Final processing of embeddings
            for phase in ['train', 'val']:
                if results['embeddings'][phase]['embeddings1']:
                    results['embeddings'][phase]['embeddings1'] = torch.cat(results['embeddings'][phase]['embeddings1'])
                    results['embeddings'][phase]['embeddings2'] = torch.cat(results['embeddings'][phase]['embeddings2'])
                    results['embeddings'][phase]['labels'] = torch.cat(results['embeddings'][phase]['labels'])

            # Log ROC curves as artifacts
            for phase in ['train', 'val']:
                if results['roc_curves'][phase]['fpr']:
                    import matplotlib.pyplot as plt
                    plt.figure()
                    plt.plot(results['roc_curves'][phase]['fpr'],
                             results['roc_curves'][phase]['tpr'],
                             label=f'{phase} (AUC = {results["metrics"][phase]["roc_auc"][-1]:.2f})')
                    plt.plot([0, 1], [0, 1], 'k--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title(f'ROC Curve - {phase}')
                    plt.legend(loc="lower right")
                    plt.savefig(f"{phase}_roc_curve.png")
                    plt.close()
                    mlflow.log_artifact(f"{phase}_roc_curve.png")

            mlflow.end_run()
            return results
        else:
            return None

    def evaluate(self,
                 loader: torch.utils.data.DataLoader,
                 criterion: nn.Module,
                 device: str = 'cuda',
                 store_embeddings: bool = True) -> Dict:
        """Evaluate the model with optional embedding storage"""
        self.eval()
        total_loss = 0.0
        all_embeddings1, all_embeddings2, all_labels = [], [], []

        with torch.no_grad():
            for img1, img2, label in loader:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                output1, output2 = self(img1, img2)
                loss = criterion(output1, output2, label)
                total_loss += loss.item()

                if store_embeddings:
                    all_embeddings1.append(output1.cpu())
                    all_embeddings2.append(output2.cpu())
                    all_labels.append(label.cpu())
                else:
                    # Just keep the last batch for metrics if not storing
                    all_embeddings1 = [output1.cpu()]
                    all_embeddings2 = [output2.cpu()]
                    all_labels = [label.cpu()]

        # Compute metrics
        embeddings1 = torch.cat(all_embeddings1).to(device)
        embeddings2 = torch.cat(all_embeddings2).to(device)
        labels = torch.cat(all_labels).to(device)

        metrics = self.compute_metrics(embeddings1, embeddings2, labels)
        metrics['loss'] = total_loss / len(loader)

        return {
            'metrics': metrics,
            'embeddings': {
                'embeddings1': all_embeddings1 if store_embeddings else [],
                'embeddings2': all_embeddings2 if store_embeddings else [],
                'labels': all_labels if store_embeddings else []
            }
        }


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


# 1. Create a simple synthetic dataset for testing
class SyntheticSiameseDataset(Dataset):
    def __init__(self, num_samples=1000, img_size=(3, 224, 224)):
        self.num_samples = num_samples
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Generate random image pairs (50% similar, 50% dissimilar)
        self.pairs = []
        for _ in range(num_samples):
            if np.random.rand() > 0.5:
                # Similar pair (same class)
                img1 = np.random.rand(*img_size).astype(np.float32)
                img2 = img1 + np.random.normal(0, 0.1, size=img_size).astype(np.float32)
                label = 1.0
            else:
                # Dissimilar pair (different class)
                img1 = np.random.rand(*img_size).astype(np.float32)
                img2 = np.random.rand(*img_size).astype(np.float32)
                label = -1.0
            self.pairs.append((img1, img2, label))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img1, img2, label = self.pairs[idx]
        return (
            self.transform(img1),
            self.transform(img2),
            torch.tensor(label, dtype=torch.float32)
        )


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# Fixed Synthetic Dataset Class
class SyntheticSiameseDataset(Dataset):
    def __init__(self, num_samples=1000, img_size=(3, 224, 224)):
        self.num_samples = num_samples
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Generate reproducible random pairs
        np.random.seed(42)
        self.pairs = []
        for _ in range(num_samples):
            if np.random.rand() > 0.5:
                # Similar pair (same class)
                base_img = np.random.rand(*img_size).astype(np.float32)
                img1 = base_img * 255
                img2 = (base_img + np.random.normal(0, 0.1, size=img_size).astype(np.float32)) * 255
                label = 1.0
            else:
                # Dissimilar pair (different class)
                img1 = np.random.rand(*img_size).astype(np.float32) * 255
                img2 = np.random.rand(*img_size).astype(np.float32) * 255
                label = -1.0
            self.pairs.append((img1, img2, label))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img1, img2, label = self.pairs[idx]

        # Convert to PIL Images first
        img1_pil = Image.fromarray(img1.transpose(1, 2, 0).astype('uint8'))
        img2_pil = Image.fromarray(img2.transpose(1, 2, 0).astype('uint8'))

        return (
            self.transform(img1_pil),
            self.transform(img2_pil),
            torch.tensor(label, dtype=torch.float32)
        )


# Test the dataset
test_dataset = SyntheticSiameseDataset(10)
img1, img2, label = test_dataset[0]
print(f"Image 1 shape: {img1.shape}")  # Should be torch.Size([3, 224, 224])
print(f"Image 2 shape: {img2.shape}")  # Should be torch.Size([3, 224, 224])
print(f"Label: {label.item()}")  # Should be 1.0 or -1.0


# Visualize a sample
def imshow(img, title=None):
    """Helper to unnormalize and show image"""
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.axis('off')


fig, (ax1, ax2) = plt.subplots(1, 2)
imshow(img1, "Image 1")
imshow(img2, f"Image 2 (Label: {label.item()})")
plt.show()

# Create data loaders
train_loader = DataLoader(SyntheticSiameseDataset(1000), batch_size=32, shuffle=True)
val_loader = DataLoader(SyntheticSiameseDataset(200), batch_size=32, shuffle=False)
test_loader = DataLoader(SyntheticSiameseDataset(200), batch_size=32, shuffle=False)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SiameseResNet().to(device)
criterion = nn.CosineEmbeddingLoss()

# Test forward pass
dummy_img1 = torch.randn(2, 3, 224, 224).to(device)
dummy_img2 = torch.randn(2, 3, 224, 224).to(device)
output1, output2 = model(dummy_img1, dummy_img2)
print(f"\nForward pass output shapes: {output1.shape}, {output2.shape}")  # Should be (2, 128)

# Test evaluation
model.eval()
with torch.no_grad():
    test_results = model.evaluate(test_loader, criterion, device)
    print(f"\nInitial test loss: {test_results['metrics']['loss']:.4f}")
    print(f"Initial test accuracy: {test_results['metrics']['accuracy']:.4f}")

# Train the model
print("\nStarting training...")
training_results = model.train_model_constructive(
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    learning_rate=0.001,
    num_epochs=5,
    device=device,
    training_mode=True
)

# Evaluate after training
model.eval()
with torch.no_grad():
    test_results = model.evaluate(test_loader, criterion, device)
    print(f"\nFinal test loss: {test_results['metrics']['loss']:.4f}")
    print(f"Final test accuracy: {test_results['metrics']['accuracy']:.4f}")
    print(f"Final test AUC: {test_results['metrics']['roc_auc']:.4f}")

# Plot training curves
if training_results:
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(training_results['metrics']['train']['loss'], label='Train')
    plt.plot(training_results['metrics']['val']['loss'], label='Validation')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(training_results['metrics']['train']['accuracy'], label='Train')
    plt.plot(training_results['metrics']['val']['accuracy'], label='Validation')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()