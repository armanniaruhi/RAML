import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
from torch.optim import Adam


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

    def train_model_constructive(self, train_loader, val_loader=None, criterion=None,
                                 learning_rate=0.001, num_epochs=10, device='cuda'):
        """
        Train the Siamese network.

        Args:
            train_loader (DataLoader): DataLoader for training data
            val_loader (DataLoader, optional): DataLoader for a validation data
            criterion (nn.Module, optional): Loss function. If None, uses ContrastiveLoss
            learning_rate (float): Learning rate for optimizer
            num_epochs (int): Number of training epochs
            devices (str): Device to use for training ('cuda' or 'cpu')
        """
        # Set device
        self.to(device)

        # Default to ContrastiveLoss if no criterion provided
        if criterion is None:
            criterion = nn.CosineEmbeddingLoss()

        # Optimizer
        optimizer = Adam(self.parameters(), lr=learning_rate)

        # Training loop
        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0

            # Progress bar
            pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')

            for batch_idx, (img1, img2, label) in enumerate(pbar):
                # Move data to a device
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                output1, output2 = self(img1, img2)

                # Compute loss
                loss = criterion(output1, output2, label)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Update running loss
                running_loss += loss.item()

                # Update progress bar
                pbar.set_postfix({'loss': running_loss / (batch_idx + 1)})

            # Validation if a validation loader is provided
            if val_loader is not None:
                val_loss = self.evaluate(val_loader, criterion, device)
                print(
                    f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}')
                return val_loss  # Return validation loss for Optuna
            else:
                print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}')
                return running_loss / len(train_loader)  # Return training loss if no validation
        return None

    def evaluate(self, val_loader, criterion, device='cuda'):
        """Evaluate the model on validation data"""
        self.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img1, img2, label in val_loader:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                output1, output2 = self(img1, img2)
                loss = criterion(output1, output2, label)
                val_loss += loss.item()
        return val_loss / len(val_loader)

