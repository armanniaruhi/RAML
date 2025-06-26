import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class SiameseNetwork(nn.Module):
    def __init__(self, loss_type="None", freeze=True):
        super(SiameseNetwork, self).__init__()
        self.loss_type = loss_type

        # Load pretrained ResNet-18 with new weights API
        weights = ResNet18_Weights.DEFAULT  # or use IMAGENET1K_V1 if you want fixed reproducibility
        self.resnet = resnet18(weights=weights)

        # Freeze all layers if requested
        if freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Replace the final fully connected layer
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256)
        )

    def forward_once(self, x):
        if "contrastive" in self.loss_type.lower():
            return self.resnet(x)
        else:
            return F.normalize(self.resnet(x), p=2, dim=1)

    def forward(self, input1, input2):
        return self.forward_once(input1), self.forward_once(input2)
