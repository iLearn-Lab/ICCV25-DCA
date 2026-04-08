"""
@author: Mingyang Liu
@contact: mingyangliu1024@gmail.com
"""
from typing import Optional
from typing import Tuple, Optional, List, Dict
import torch.nn as nn
import torch
import torch.nn.functional as F



class SupConResNetMultiBottlenecks(nn.Module):
    r"""SupCon ResNet.

    Args:
        in_features (int): Dimension of input features
        num_classes (int): Number of classes
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: 128

    Shape:
        - Inputs: :math:`(minibatch, F_in)` where F_in = `in_features`.
        - Output: :math:`(C, minibatch, F_out)` where C = `num_classes`, F_out = `bottleneck_dim`.
    """

    def __init__(self, in_features: int, num_classes: int, bottleneck_dim: Optional[int] = 128, pool_layer=None):
        super(SupConResNetMultiBottlenecks, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = bottleneck_dim
        if pool_layer is None:
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        else:
            self.pool_layer = pool_layer
        # self.head = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(in_features, bottleneck_dim),
        #     nn.BatchNorm1d(bottleneck_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(bottleneck_dim, bottleneck_dim),
        #     nn.BatchNorm1d(bottleneck_dim),
        #     nn.ReLU(),
        #     nn.Linear(bottleneck_dim, num_classes)
        # )
        self.bottleneck  = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(in_features, in_features),
                nn.ReLU(inplace=True),
                nn.Linear(in_features, bottleneck_dim)) 
                for i in range(self.num_classes)]
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        f = self.pool_layer(inputs)
        f_normalize_list = []
        f_list = []
        for i in range(self.num_classes):
            f_normalize_list.append(F.normalize(self.bottleneck[i](f), dim=1))
            f_list.append(self.bottleneck[i](f))
        f_normalize = torch.stack(f_normalize_list,dim=0)
        f = torch.stack(f_list,dim=0)
        if self.training:
            return f_normalize, f
        else:
            return f_normalize

    
    def get_parameters(self) -> List[Dict]:
        return [{"params": self.bottleneck.parameters(), "lr": 1.}]
    
class ImageClassifierHead(nn.Module):
    r"""Classifier Head for MCD.

    Args:
        in_features (int): Dimension of input features
        num_classes (int): Number of classes
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: 1024

    Shape:
        - Inputs: :math:`(minibatch, F)` where F = `in_features`.
        - Output: :math:`(minibatch, C)` where C = `num_classes`.
    """

    def __init__(self, in_features: int, num_classes: int, bottleneck_dim: Optional[int] = 1024, pool_layer=None):
        super(ImageClassifierHead, self).__init__()
        self.num_classes = num_classes
        if pool_layer is None:
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        else:
            self.pool_layer = pool_layer
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, num_classes)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool_layer(inputs.detach()))
    

class SupConResNetBottleneck(nn.Module):
    r"""SupCon ResNet.

    Args:
        in_features (int): Dimension of input features
        num_classes (int): Number of classes
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: 128

    Shape:
        - Inputs: :math:`(minibatch, F_in)` where F_in = `in_features`.
        - Output: :math:`(C, minibatch, F_out)` where C = `num_classes`, F_out = `bottleneck_dim`.
    """

    def __init__(self, in_features: int, num_classes: int, bottleneck_dim: Optional[int] = 128, pool_layer=None):
        super(SupConResNetMultiBottlenecks, self).__init__()
        self.num_classes = num_classes
        if pool_layer is None:
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        else:
            self.pool_layer = pool_layer
        # self.head = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(in_features, bottleneck_dim),
        #     nn.BatchNorm1d(bottleneck_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(bottleneck_dim, bottleneck_dim),
        #     nn.BatchNorm1d(bottleneck_dim),
        #     nn.ReLU(),
        #     nn.Linear(bottleneck_dim, num_classes)
        # )
        self.bottleneck  = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(in_features, in_features),
                nn.ReLU(inplace=True),
                nn.Linear(in_features, bottleneck_dim)) 
                for i in range(self.num_classes)]
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        f = self.pool_layer(inputs)
        f_list = []
        for i in range(self.num_classes):
            f_list.append(F.normalize(self.bottleneck[i](f), dim=1))
        f = torch.stack(f_list,dim=0)
        return f