import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
from torch.optim import Adam
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_curve, roc_auc_score)
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
        # Freeze all ResNet layers
        for param in self.resnet.parameters():
            param.requires_grad = False

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
            # Set relative path for local MLflow tracking directory
            mlflow.set_tracking_uri("file:./../../mlruns")
            mlflow.set_experiment(experiment_name)
            # Start logging
            with mlflow.start_run():
                mlflow.log_param("learning_rate", learning_rate)
                mlflow.log_param("num_epochs", num_epochs)

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