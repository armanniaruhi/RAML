import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, roc_auc_score)
from typing import Dict, Optional, Tuple
import tqdm
from src.ml.model_utils import EarlyStopping
from pytorch_metric_learning.miners import PairMarginMiner

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
                    patience: int = 7,
                    scheduler_type: str = 'reduce_on_plateau',  # New: scheduler selection
                    save_path: str = 'best_model.pth') -> Dict[str, list]:
        
        # Initialize optimizer (now with AdamW as default)
        if optimizer is None:
            optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
        
        if criterion is None:
            raise ValueError("Criterion (loss function) must be provided")

        # Initialize scheduler based on type
        if scheduler_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.1, 
                patience=3, 
                verbose=True
            )
        elif scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=num_epochs,
                eta_min=1e-6,
                last_epoch=-1
            )
        elif scheduler_type == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=1e-3,
                steps_per_epoch=len(train_loader),
                epochs=num_epochs
            )
        else:
            scheduler = None

        self.to(device)
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        train_loss_history = []
        val_loss_history = []
        val_accuracy_history = []
        pbar = tqdm.tqdm(range(num_epochs), desc="Training")

        for epoch in pbar:
            self.train()
            train_loss = 0.0

            for idx, (imgs, labels) in enumerate(train_loader):
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                embeddings = self(imgs)
                miner = PairMarginMiner(pos_margin=0.2, neg_margin=0.8)
                hard_pairs = miner(embeddings, labels)
                loss = criterion(embeddings, labels, hard_pairs)
                loss.backward()
                optimizer.step()
                
                # Step per batch for OneCycleLR
                if scheduler_type == 'onecycle':
                    scheduler.step()
                
                train_loss += loss.item()
                pbar.set_postfix({
                    'batch progress': f'{idx + 1}/{len(train_loader)}',
                    'batch loss': f'{train_loss/(idx + 1)}',
                    'lr': optimizer.param_groups[0]['lr']  # Show current learning rate
                })

            avg_train_loss = train_loss / len(train_loader)
            train_loss_history.append(avg_train_loss)

            # Validation phase
            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                val_acc = 0.0
                with torch.no_grad():
                    for imgs, labels in val_loader:
                        imgs, labels = imgs.to(device), labels.to(device)
                        embeddings = self(imgs)
                        loss = criterion(embeddings, labels)
                        val_loss += loss.item()

                        acc = self.calculate_pairwise_accuracy(embeddings, labels)
                        val_acc += acc

                avg_val_loss = val_loss / len(val_loader)
                avg_val_acc = val_acc / len(val_loader)
                val_loss_history.append(avg_val_loss)
                val_accuracy_history.append(avg_val_acc)

                if scheduler_type == 'reduce_on_plateau':
                    scheduler.step(avg_val_loss)
                elif scheduler_type == 'cosine':
                    scheduler.step()

                early_stopping(avg_val_loss, self)
            else:
                if scheduler_type == 'cosine':
                    scheduler.step()
                early_stopping(avg_train_loss, self)

            pbar.set_postfix({
                'train_loss': f'{avg_train_loss:.6f}',
                'val_loss': f'{avg_val_loss:.6f}' if val_loader else 'N/A',
                'val_acc': f'{avg_val_acc:.4f}' if val_loader else 'N/A',
                'lr': optimizer.param_groups[0]['lr']
            })

            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        if early_stopping.best_model_state is not None:
            self.load_state_dict(early_stopping.best_model_state)

        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)
        print(f"Model saved to {save_path}")

        return {
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history,
            'val_accuracy_history': val_accuracy_history
        }
        
        
    def calculate_pairwise_accuracy(embeddings, labels, threshold=0.5):
        # Calculate pairwise distances
        cos = nn.CosineSimilarity(dim=1)
        total = 0
        correct = 0
        for i in range(0, len(embeddings), 2):
            if i + 1 >= len(embeddings): break
            emb1, emb2 = embeddings[i], embeddings[i + 1]
            label1, label2 = labels[i], labels[i + 1]
            sim = cos(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
            pred_similar = sim > threshold
            true_similar = label1 == label2
            correct += (pred_similar == true_similar)
            total += 1
        return correct / total if total > 0 else 0