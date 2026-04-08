"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""
from typing import Optional
import torch
import torch.nn as nn
from tllib.modules.classifier import Classifier as ClassifierBase



class BSPLoss(nn.Module):
    def __init__(self):
        super(SpectralDebiasingLoss, self).__init__()

    def forward(self, f_1, f_2, f_3=None, batch=2):
        if batch == 2:
            _, s_1, _ = torch.svd(f_1)
            if f_2.size(0) == 0:
                loss = torch.pow(s_1[0], 2)
            else:
                _, s_2, _ = torch.svd(f_2)
                loss = s_1.mean() + s_2.mean()
        elif batch == 3:
            _, s_1, _ = torch.svd(f_1)
            _, s_2, _ = torch.svd(f_2)
            _, s_3, _ = torch.svd(f_3)
            loss = torch.pow(s_1[0], 2) + 0.5*(torch.pow(s_2[0], 2)+torch.pow(s_3[0], 2))
        return loss



class SpectralDebiasingLoss(nn.Module):
    def __init__(self):
        super(SpectralDebiasingLoss, self).__init__()

    def forward(self, f_1, f_2, f_3,):
        _, s_1, _ = torch.svd(f_1)
        _, s_2, _ = torch.svd(f_2)
        _, s_3, _ = torch.svd(f_3)
        loss = torch.pow(s_1[0], 2) + 0.5*(torch.pow(s_2[0], 2)+torch.pow(s_3[0], 2))
        return loss


class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        bottleneck = nn.Sequential(
            # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            # nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)
