import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        # Load ResNet18 with pretrained weights
        self.cnn1 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Store the original first conv layer
        original_first_conv = self.cnn1.conv1
        
        # Replace with a 1-channel input conv layer
        self.cnn1.conv1 = nn.Conv2d(
            1,  # Single channel input
            original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=original_first_conv.stride,
            padding=original_first_conv.padding,
            bias=original_first_conv.bias
        )
        
        # Initialize weights by averaging RGB channels
        with torch.no_grad():
            self.cnn1.conv1.weight = nn.Parameter(original_first_conv.weight.mean(dim=1, keepdim=True))
        
        # Freeze the backbone
        for param in self.cnn1.parameters():
            param.requires_grad = False
        
        # Replace the FC layer with identity
        self.cnn1.fc = nn.Identity()
        
        # Custom head
        self.fc1 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
        )
    
    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output
        #return F.normalize(output, p=2, dim=1) #JUST FOR COSINE-BASED ALGORITHMS
    
    def forward(self, input1, input2):
        return self.forward_once(input1), self.forward_once(input2)