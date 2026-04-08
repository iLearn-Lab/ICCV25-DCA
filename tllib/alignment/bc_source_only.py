"""
@author: Mingyang Liu
@contact: mingyangliu1024@gmail.com
"""
from typing import Optional
import torch.nn as nn
import torch

class ImageClassifierHeadMode4(nn.Module):
    r"""Classifier Head for multi-label decouple.

    Model:
        num_classes*(bottleneck,head)

    Args:
        in_features (int): Dimension of input features
        num_classes (int): Number of classes
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: 1024

    Shape:
        - Inputs: :math:`(minibatch, F)` where F = `in_features`.
        - Output: :math:`(minibatch, C)` where C = `num_classes`.
    """


    def __init__(self, in_features: int, bottleneck_dim: Optional[int] = 1024, pool_layer=None, num_classes=1):
        super(ImageClassifierHeadMode4, self).__init__()
        if pool_layer is None:
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        else:
            self.pool_layer = pool_layer
        self.bottleneck = nn.Identity()
        self.head  = nn.Linear(in_features, num_classes)
            

    def forward(self, inputs: torch.Tensor, get_f=False) -> torch.Tensor:
        f = self.pool_layer(inputs)
        if get_f:
            return torch.squeeze(self.head(f),dim=1), f # b*1
        else:
            return torch.squeeze(self.head(f),dim=1)