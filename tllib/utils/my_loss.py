from typing import Optional, List, Dict, Tuple, Callable
import torch.nn as nn
import torch.nn.functional as F
import torch

class FocalLoss(nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        # 返回
        self.bce =  nn.BCEWithLogitsLoss(reduction=None)
    
    def forward(self, output, target):
        logp = self.bce(output, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()
    
def focalLoss(y_pred, y_true, gamma=1, detach=None):
    # 
    assert y_pred.shape == y_true.shape
    if detach:
        pt = torch.exp(-F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none'))
        sample_weight = ((1 - pt) ** gamma).detach()
        return F.binary_cross_entropy_with_logits(y_pred, y_true, weight=sample_weight,reduction='mean')
    else:
        pt = torch.exp(-F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='none'))
        sample_weight = ((1 - pt) ** gamma)
        return torch.mean(sample_weight * F.binary_cross_entropy_with_logits(y_pred, y_true,reduction='none'))

# def mean_distance_loss(y_pred, y_true):

