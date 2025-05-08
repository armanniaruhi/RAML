import torch
from .model import EmbeddingNet, SiameseNet
from .losses import ContrastiveLoss
import torch.nn as nn

def train(dataloader, loss_func, epoch_num= 10, lr = 0.001):
    # Model
    embedding_net = EmbeddingNet()
    model = SiameseNet(embedding_net)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epoch_num):
        for img1, img2, label in dataloader:
            out1, out2 = model(img1, img2)
            # === Uncomment ONE ===
            loss = loss_func(out1, out2, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")