import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, roc_auc_score)
from typing import Dict, Optional, Tuple
import tqdm
from src.ml.model_utils import EarlyStopping


class SiameseResNet(nn.Module):
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 512):
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
        """
        super(SiameseResNet, self).__init__()
        # Load pretrained ResNet50 with the latest weights
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        # Remove the final fully connected layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        # Freeze all ResNet layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(2048, hidden_dim),  # First fully connected layer
            nn.ReLU(),  # Activation function
            nn.Linear(hidden_dim, embedding_dim)  # Second fully connected layer
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)

        Returns:
            torch.Tensor: Output embedding of shape (batch_size, 128)
        """
        x = self.resnet(x)
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        output = self.fc(x)
        return output

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

            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                embeddings = self(imgs)
                loss = criterion(embeddings, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            train_loss_history.append(avg_train_loss)

            avg_val_loss = None
            val_accuracy = None

            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                all_outputs = []
                all_labels = []

                with torch.no_grad():
                    for imgs, labels in val_loader:
                        imgs, labels = imgs.to(device), labels.to(device)
                        embeddings = self(imgs)
                        loss = criterion(embeddings, labels)
                        val_loss += loss.item()

                        # Compute metrics using embeddings
                        if hasattr(self, 'compute_metrics'):
                            # Create dummy pairwise inputs: embeddings1 and embeddings2
                            half = embeddings.size(0) // 2
                            if half > 0:
                                out1 = embeddings[:half]
                                out2 = embeddings[half:2*half]
                                lbls = labels[:half]
                                metrics = self.compute_metrics(out1, out2, lbls)
                                val_accuracy = metrics['accuracy']

                avg_val_loss = val_loss / len(val_loader)
                val_loss_history.append(avg_val_loss)

                early_stopping(avg_val_loss, self)
            else:
                early_stopping(avg_train_loss, self)

            # Set progress bar postfix
            pbar.set_postfix({
                'train_loss': f'{avg_train_loss:.6f}',
                'val_loss': f'{avg_val_loss:.6f}' if avg_val_loss is not None else 'N/A',
                'val_acc': f'{val_accuracy:.4f}' if val_accuracy is not None else 'N/A'
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