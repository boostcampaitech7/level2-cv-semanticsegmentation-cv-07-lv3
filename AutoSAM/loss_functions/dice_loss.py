import torch
import numpy as np
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, eps=0.0001):
        super(DiceLoss, self).__init__()
        self.eps = eps
        
    def forward(self, probs, targets):
        # probs = torch.sigmoid(logits)
        
        # Flatten
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        total = probs.sum() + targets.sum()
        
        dice = (2.0 * intersection + self.eps) / (total + self.eps)
        return 1.0 - dice
