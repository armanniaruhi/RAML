import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class ContrastiveLoss(nn.Module):
    """Contrastive loss function.
    
    Based on the concept that similar pairs should have close embeddings while 
    dissimilar pairs should have distant embeddings.
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        
    def forward(self, output1, output2, label):
        """
        Args:
            output1: First embedding vector
            output2: Second embedding vector
            label: Binary label (1 for similar pairs, 0 for dissimilar pairs)
        """
        dist = F.pairwise_distance(output1, output2)
        loss = (label) * torch.pow(dist, 2) + \
               (1 - label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        return loss.mean()

class MSLoss(nn.Module):
    """Multi-Similarity Loss.
    
    Considers multiple types of similarities between samples: self-similarity,
    positive-pair similarity, and negative-pair similarity.
    """
    def __init__(self, alpha=2.0, beta=50.0, lambda_=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda_ = lambda_
        
    def forward(self, features, labels):
        """
        Args:
            features: Batch of embedding vectors (N x D)
            labels: Corresponding labels (N)
        """
        batch_size = features.size(0)
        sim_mat = torch.matmul(features, features.t())
        
        # Get positive and negative mask
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.t()).float()
        neg_mask = (labels != labels.t()).float()
        
        # Positive pairs
        pos_exp = torch.exp(-self.alpha * (sim_mat - self.lambda_))
        pos_loss = torch.log(1 + torch.sum(pos_exp * pos_mask, dim=1))
        
        # Negative pairs
        neg_exp = torch.exp(self.beta * (sim_mat - self.lambda_))
        neg_loss = torch.log(1 + torch.sum(neg_exp * neg_mask, dim=1))
        
        loss = torch.mean(pos_loss + neg_loss)
        return loss

class HistogramLoss(nn.Module):
    """Histogram Loss.
    
    Compares distributions of similar and dissimilar pairs using histogram-based 
    approximation.
    """
    def __init__(self, num_steps=100):
        super().__init__()
        self.num_steps = num_steps
        
    def forward(self, features, labels):
        """
        Args:
            features: Batch of embedding vectors (N x D)
            labels: Corresponding labels (N)
        """
        # Compute pairwise distances
        dist_mat = torch.cdist(features, features, p=2)
        
        # Get positive and negative masks
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.t()).float()
        neg_mask = (labels != labels.t()).float()
        
        # Separate positive and negative distances
        pos_dists = dist_mat[pos_mask.bool()]
        neg_dists = dist_mat[neg_mask.bool()]
        
        # Create histograms
        min_dist = min(pos_dists.min().item(), neg_dists.min().item())
        max_dist = max(pos_dists.max().item(), neg_dists.max().item())
        step = (max_dist - min_dist) / self.num_steps
        
        pos_hist = torch.histc(pos_dists, bins=self.num_steps, min=min_dist, max=max_dist)
        neg_hist = torch.histc(neg_dists, bins=self.num_steps, min=min_dist, max=max_dist)
        
        # Normalize histograms
        pos_hist = pos_hist / pos_hist.sum()
        neg_hist = neg_hist / neg_hist.sum()
        
        # Compute cumulative distributions
        pos_cdf = torch.cumsum(pos_hist, dim=0)
        neg_cdf = torch.cumsum(neg_hist, dim=0)
        
        # Compute loss as the area between CDFs
        loss = torch.mean(torch.abs(pos_cdf - neg_cdf))
        return loss

# TripletLoss (keep existing implementation)
triplet_loss_fn = nn.TripletMarginLoss(margin=1.0)

# BCE Loss (keep existing implementation)
bce_loss_fn = nn.BCELoss()

