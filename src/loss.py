import torch
import torch.nn as nn
import torch.nn.functional as F

class LossFactory:
    """Factory class for creating loss functions"""
    @staticmethod
    def get_loss(loss_config):
        loss_name = loss_config['NAME'].lower()
        
        if loss_name == 'bce':
            return nn.BCEWithLogitsLoss()
        elif loss_name == 'dice':
            return DiceLoss()
        elif loss_name == 'combined':
            return CombinedLoss(
                losses=[nn.BCEWithLogitsLoss(), DiceLoss()],
                weights=loss_config['WEIGHTS']
            )
        elif loss_name == 'boundary':
            return CombinedLoss(
                losses=[nn.BCEWithLogitsLoss(), DiceLoss(), BoundaryLoss()],
                weights=loss_config['WEIGHTS']
            )
        else:
            raise NotImplementedError(f"Loss {loss_name} not implemented")

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        
        # Flatten
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        total = probs.sum() + targets.sum()
        
        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        
        return 1.0 - dice

class CombinedLoss(nn.Module):
    def __init__(self, losses, weights=None):
        super(CombinedLoss, self).__init__()
        self.losses = losses
        self.weights = weights or [1.0] * len(losses)
        
    def forward(self, inputs, targets, dist_maps=None):
        total_loss = 0

        for loss, weight in zip(self.losses, self.weights):
            if (isinstance(loss, BoundaryLoss)):
                if dist_maps is None:
                    raise ValueError("BoundaryLoss requires 'dist_maps' as an input.")
                total_loss += weight * loss(inputs, dist_maps)
            else:
                total_loss += weight * loss(inputs, targets)

        return total_loss

class BoundaryLoss(nn.Module):
    def __init__(self):
        super(BoundaryLoss, self).__init__()
    
    def forward(self, logits, dist_maps):
        probs = torch.sigmoid(logits)
        
        multipled = probs * dist_maps
        loss = multipled.mean()

        return loss