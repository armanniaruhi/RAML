import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss function.
    This loss encourages similar pairs to have closer embeddings and dissimilar pairs to have distant embeddings.
    
    Based on the paper "Dimensionality Reduction by Learning an Invariant Mapping"
    by Hadsell, Chopra, and LeCun (2006).
    """
    
    def __init__(self, margin=2.0):
        """
        Initialize ContrastiveLoss with a margin.
        
        Args:
            margin (float): Minimum distance margin between dissimilar pairs.
                          Pairs with distance > margin do not contribute to the loss.
                          Default: 2.0
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """
        Calculate the contrastive loss between pairs of embeddings.
        
        Args:
            output1 (torch.Tensor): First embedding tensor of shape (batch_size, embedding_dim)
            output2 (torch.Tensor): Second embedding tensor of shape (batch_size, embedding_dim)
            label (torch.Tensor): Binary labels indicating if pairs are similar (0) or dissimilar (1)
                                Shape: (batch_size,)
        
        Returns:
            torch.Tensor: Scalar loss value
            
        Formula:
            For similar pairs (label=1):
                Loss = d²
                where d is the Euclidean distance between embeddings
            
            For dissimilar pairs (label=0):
                Loss = max(0, margin - d)²
                encouraging distance to be at least equal to margin
        """
        # Calculate Euclidean distance between pairs of embeddings
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)

        # Compute contrastive loss:
        loss = torch.mean(
            # Term for similar pairs (label=1): minimize distance
            label * torch.pow(euclidean_distance, 2) +
            
            # Term for dissimilar pairs (label=0): maximize distance up to margin
            (1 - label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        
        return loss