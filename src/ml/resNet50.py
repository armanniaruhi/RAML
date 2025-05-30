import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, roc_auc_score)
from typing import Dict, Optional, Tuple
import tqdm
from src.ml.model_utils import EarlyStopping

class SiameseResNet(nn.Module):
    def __init__(self, embedding_dim: int = 256, hidden_dim: list[int] = [1024, 512]):
        """
        SiameseResNet model using a frozen ResNet50 backbone and a customizable fully connected head.

        Args:
            embedding_dim (int): Final embedding dimension (default: 128)
            hidden_dim (list[int]): List of hidden layer sizes (default: [1024, 512, 256])
        """
        super(SiameseResNet, self).__init__()

        # Load pretrained ResNet50 backbone and remove the final FC layer
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # (B, 2048, 1, 1)

        # Freeze ResNet backbone
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Build fully connected layers dynamically
        fc_layers = []
        in_dim = 2048   # Resnet50
        in_dim = 512    # Resnet18
        for h_dim in hidden_dim:
            fc_layers.append(nn.Linear(in_dim, h_dim))
            fc_layers.append(nn.ReLU())
            in_dim = h_dim
        fc_layers.append(nn.Linear(in_dim, embedding_dim))  # Final layer to embedding_dim

        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.resnet(x)  # Shape: (B, 2048, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to (B, 2048)
        x = self.fc(x)  # Final shape: (B, embedding_dim)
        return x


    def compute_metrics(self,
                        outputs1: torch.Tensor,
                        outputs2: torch.Tensor,
                        labels: torch.Tensor,
                        threshold: float = 0.5) -> Dict[str, float]:
        """Compute classification metrics including ROC/AUC

        Args:
            outputs1: Embeddings from first image
            outputs2: Embeddings from second image
            labels: Ground truth labels (1 for similar, 0 for dissimilar)
            threshold: Threshold for binary classification

        Returns:
            Dictionary containing metrics and ROC curve data
        """
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

    def train_model(self,
                    train_loader: torch.utils.data.DataLoader,
                    val_loader: Optional[torch.utils.data.DataLoader] = None,
                    criterion: Optional[nn.Module] = None,
                    num_epochs: int = 10,
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    experiment_name: str = 'SiameseResNet',
                    tuning_mode: bool = True,
                    patience: int = 7
                    ) -> Dict[str, list]:
        """Train the Siamese network.

        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            criterion: Loss function
            num_epochs: Number of training epochs
            optimizer: Optimization algorithm
            device: Device to train on (cuda or cpu)
            experiment_name: Name for tracking
            tuning_mode: Whether in hyperparameter tuning mode
            patience: Patience for early stopping

        Returns:
            Dictionary containing training history
        """
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters())

        if criterion is None:
            raise ValueError("Criterion (loss function) must be provided")

        self.to(device)

        early_stopping = EarlyStopping(patience=patience, verbose=True)
        train_loss_history = []
        val_loss_history = []

        pbar = tqdm.tqdm(range(num_epochs), desc="Training")

        for epoch in pbar:
            self.train()
            train_loss = 0.0

            for idx, (imgs, labels) in enumerate(train_loader):
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                embeddings = self(imgs)
                loss = criterion(embeddings, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                pbar.set_postfix({
                    'batch idx': f'{idx + 1}/{len(train_loader)}',
                    'batch_loss': f'{train_loss/(idx + 1):.6f}',
                })
                avg_val_loss = None

            avg_train_loss = train_loss / len(train_loader)
            train_loss_history.append(avg_train_loss)

            if val_loader is not None:
                self.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for imgs, labels in val_loader:
                        imgs, labels = imgs.to(device), labels.to(device)
                        embeddings = self(imgs)
                        loss = criterion(embeddings, labels)
                        val_loss += loss.item()


                avg_val_loss = val_loss / len(val_loader)
                val_loss_history.append(avg_val_loss)

                early_stopping(avg_val_loss, self)
            else:
                early_stopping(avg_train_loss, self)

            # Set progress bar postfix
            pbar.set_postfix({
                'train_loss': f'{avg_train_loss:.6f}',
                'val_loss': f'{avg_val_loss:.6f}' if avg_val_loss is not None else 'N/A'
            })

            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        if early_stopping.best_model_state is not None:
            self.load_state_dict(early_stopping.best_model_state)

        return {
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
        }