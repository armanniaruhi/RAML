import torch
import torch.nn as nn

# Sample encoder network
class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)


# Siamese network: takes two inputs and returns their embeddings
class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super().__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        return self.embedding_net(x1), self.embedding_net(x2)



