import torch
import torch.nn as nn
from pytorch_metric_learning.losses import MultiSimilarityLoss as msLoss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_positive = (1 - label) * torch.pow(euclidean_distance, 2)
        loss_negative = label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        total_loss = torch.mean(loss_positive + loss_negative)
        return total_loss


class MultiSimilarityLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=50.0, base=0.5):
        super().__init__()
        self.loss_fn = msLoss(alpha=alpha, beta=beta, base=base)

    def forward(self, embeddings, labels):
        return self.loss_fn(embeddings, labels)
