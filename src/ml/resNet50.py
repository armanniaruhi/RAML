import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
from torch.optim import AdamW
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_curve, roc_auc_score)
import mlflow
import mlflow.pytorch
from typing import Dict, Optional
from src.ml.model_utils import EarlyStopping


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
        # Freeze all ResNet layers
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.fc = nn.Sequential(
            nn.Linear(2048, 512),            # First fully connected layer
            nn.ReLU(),                                     # Activation function
            nn.Linear(512, 128),     # Second fully connected layer (outputs embedding)
        )

    def forward_one(self, x):
        x = self.resnet(x)
        x = x.view(x.size()[0], -1)
        output = self.fc(x)
        return output   # Emmedding of shape (batch_size, 128)
    
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
                                 num_epochs: int = 10,
                                 optimizer: Optional[nn.Module] = torch.optim.Adam,
                                 device: torch.device = 'cuda',
                                 experiment_name: str = 'SiameseResNet',
                                 tuning_mode: bool = True,
                                 patience: int = 7
                                 ) -> Optional[Dict]:
        """
        Train or evaluate the model with optional tracking

        Args:
            tuning_mode: If True, stores metrics, embeddings and logs to MLflow.
                          If False, only computes metrics without storing.
            patience: Number of epochs to wait before early stopping
        """

        # Initialize early stopping
        early_stopping = EarlyStopping(patience=patience, verbose=True)

        counter = []
        train_loss_history = []
        val_loss_history = []
        iteration_number = 0

        # Iterate through the epochs
        for epoch in range(num_epochs):
            # Training phase
            self.train()
            train_loss = 0.0
            train_n_batches = 0

            # Iterate over training batches
            for i, (img0, img1, label) in enumerate(train_loader, 0):
                # Move data to device
                img0, img1 = img0.to(device), img1.to(device)
                label = label.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                output1, output2 = self(img0, img1)

                # Calculate loss
                loss_contrastive = criterion(output1, output2, label)

                # Backward pass and optimize
                loss_contrastive.backward()
                optimizer.step()

                # Accumulate loss
                train_loss += loss_contrastive.item()
                train_n_batches += 1

                # Log training progress
                if i % 5 == 0:
                    train_loss_history.append(loss_contrastive.item())
                    print(f"Epoch {epoch} - Iteration {i} - Training Loss: {loss_contrastive.item():.4f}")

            # Calculate average training loss for the epochs
            avg_train_loss = train_loss / train_n_batches

            # Validation phase
            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                val_n_batches = 0

                with torch.no_grad():
                    for img0, img1, label in val_loader:
                        # Move data to device
                        img0, img1 = img0.to(device), img1.to(device)
                        label = label.to(device)

                        # Forward pass
                        output1, output2 = self(img0, img1)

                        # Calculate loss
                        loss_contrastive = criterion(output1, output2, label)

                        # Accumulate loss
                        val_loss += loss_contrastive.item()
                        val_n_batches += 1

                # Calculate average validation loss
                avg_val_loss = val_loss / val_n_batches
                val_loss_history.append(avg_val_loss)

                print(f"Epoch {epoch} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

                # Early stopping check using validation loss
                early_stopping(avg_val_loss, self)
            else:
                # If no validation loader, use training loss for early stopping
                early_stopping(avg_train_loss, self)

            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        # Load the best model weights
        self.load_state_dict(early_stopping.best_model_state)

        return {
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
            'counter': counter
        }

    def train_model_ms(self, train_loader, val_loader=None,
                       criterion=None, num_epochs=10,
                       optimizer=None, device='cuda',
                       patience=5, tuning_mode=True):
        '''
        Train the model using MultiSimilarityLoss on batches of (image, label) pairs.

        Args:
            train_loader: DataLoader returning (image, label) tuples.
            criterion: Multi-Similarity Loss function.
            num_epochs: Number of training epochs.
            optimizer: Optimizer for model parameters.
            device: Training device ('cuda' or 'cpu').
            patience: Early stopping patience (number of epochs).
        '''

        self.train()
        early_stopping = EarlyStopping(patience=patience, verbose=True)

        for epoch in range(num_epochs):  # Epoch loop (e.g. 10 Epochs)
            train_loss = 0.0
            for imgs, labels in train_loader:  # Batch loop (e.g. 32 Img)
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()

                # Embeddings calculation (128-dim vector)
                embeddings = self.forward_one(imgs)

                # Loss calculation (MS Loss automatically handles the pairwise comparisons)
                loss = criterion(embeddings, labels)

                # Gradient calculation
                loss.backward()

                # Parameter update
                optimizer.step()

                # Loss summation
                train_loss += loss.item()

            avg_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch} - MS Training Loss: {avg_loss:.4f}")

            early_stopping(avg_loss, self)
            if early_stopping.early_stop:
                break

        self.load_state_dict(early_stopping.best_model_state)
