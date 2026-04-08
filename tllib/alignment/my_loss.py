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
    
def focalLoss(y_pred, y_true, gamma=1):
    # y_pred is the logits before Sigmoid
    assert y_pred.shape == y_true.shape
    bce = nn.BCEWithLogitsLoss(reduction=None)
    pt = torch.exp(-bce(y_pred, y_true, reduction='none')).detach()
    sample_weight = (1 - pt) ** gamma
    return bce(y_pred, y_true, weight=sample_weight)