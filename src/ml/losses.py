import torch.nn as nn
import torch
import torch.nn.functional as F

'''  TO DO List
    1. implement Contrastive Loss
    2. implement MS Loss
    3. implement Histogram Loss
'''

# Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        dist = F.pairwise_distance(output1, output2)
        loss = (label) * torch.pow(dist, 2) + \
               (1 - label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        return loss.mean()

# TripletLoss
triplet_loss_fn = nn.TripletMarginLoss(margin=1.0)

# BCE Loss
bce_loss_fn = nn.BCELoss()

