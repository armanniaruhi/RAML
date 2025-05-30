import torch
import torch.nn as nn
from torchvision.models import resnet18

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        # Using ResNet18 as the backbone with 3-channel input
        self.cnn1 = resnet18(pretrained=True)  # Using pretrained weights
        
        # Remove the original fully connected layer of ResNet18
        self.cnn1.fc = nn.Identity()
        
        # Setting up your custom Fully Connected Layers
        # ResNet18 outputs 512 features
        self.fc1 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
        )
        
    def forward_once(self, x):
        # Forward pass through ResNet18
        output = self.cnn1(x)
        # Flatten the output
        output = output.view(output.size()[0], -1)
        # Forward pass through custom FC layers
        output = self.fc1(output)
        return output
    
    def forward(self, input1, input2):
        # Forward pass for both inputs
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2