import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from transformers import ResNetForImageClassification


class SiameseNetwork(nn.Module):
    def __init__(self, model_path = "microsoft/resnet-50", stages_resnet=3, fc_dim=512, out_dim = 256):
        super(SiameseNetwork, self).__init__()
        resnet_cnn = ResNetForImageClassification.from_pretrained(model_path).resnet

        self.pretrained_cnn = nn.Sequential(*([resnet_cnn.embedder]+[resnet_cnn.encoder.stages[s] for s in range(stages_resnet)]) )
        for param in self.pretrained_cnn.parameters():
            param.requires_grad_(False); 

        # self.siamese_cnn = nn.Sequential(*[resnet_cnn.encoder.stages[s] for s in range(stages_resnet,len(resnet_cnn.encoder.stages))])
        self.siamese_cnn = nn.Sequential(*[resnet_cnn.encoder.stages[stages_resnet].layers[0], nn.AdaptiveAvgPool2d(output_size=(1, 1))])

        self.cnn = nn.Sequential(self.pretrained_cnn, self.siamese_cnn)

        self.fc1 = nn.Sequential(
            nn.Linear(2048, fc_dim),
            nn.ReLU(inplace=True),

            nn.Linear(fc_dim, fc_dim),
            nn.ReLU(inplace=True),

            nn.Linear(fc_dim, out_dim))

    def forward_once(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
        