"""
@author: Mingyang Liu
@contact: mingyangliu1024@gmail.com
"""
from typing import Optional
import torch.nn as nn
import torch


class ImageClassifierHeadMode1(nn.Module):
    r"""Classifier Head for multi-label decouple.

    Model:
        1*bottleneck + num_classes*head

    Args:
        in_features (int): Dimension of input features
        num_classes (int): Number of classes
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: 256

    Shape:
        - Inputs: :math:`(minibatch, F)` where F = `in_features`.
        - Output: :math:`(minibatch, C)` where C = `num_classes`.
    """

    def __init__(self, in_features: int, num_classes: int, bottleneck_dim: Optional[int] = 256, pool_layer=None):
        super(ImageClassifierHeadMode1, self).__init__()
        self.num_classes = num_classes
        if pool_layer is None:
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        else:
            self.pool_layer = pool_layer
        self.bottleneck = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        self.head  = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(bottleneck_dim, 1)) 
                for i in range(num_classes)]
        )

    def forward(self, inputs: torch.Tensor, train_mode='train_g', current_class=None) -> torch.Tensor:
        if train_mode == 'train_f':

            with torch.no_grad():
                f = self.bottleneck(self.pool_layer(inputs)) # b*bottleneck_dim
            return torch.squeeze(self.head[current_class](f),dim=1) # b*1
        
        else:
            f = self.bottleneck(self.pool_layer(inputs)) # b*bottleneck_dim
            output_list = []
            for i in range(self.num_classes):
                output_list.append(torch.squeeze(self.head[i](f),dim=1)) # b*1
            output = torch.stack(output_list,dim=1) # b*6
            return output
        

    def set_bottleneck_eval(self):
        self.bottleneck.requires_grad = False


class ImageClassifierHeadMode2(nn.Module):
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


    def __init__(self, in_features: int, num_classes: int, bottleneck_dim: Optional[int] = 1024, pool_layer=None):
        super(ImageClassifierHeadMode2, self).__init__()
        self.num_classes = num_classes
        if pool_layer is None:
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        else:
            self.pool_layer = pool_layer
        self.bottleneck = nn.ModuleList(
            [nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
                nn.ReLU()) 
                for i in range(num_classes)]
        )
        self.head  = nn.ModuleList(
            [nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(bottleneck_dim, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
                nn.ReLU(),
                nn.Linear(bottleneck_dim, 1)) 
                for i in range(num_classes)]
        )

    def forward(self, inputs: torch.Tensor, train_mode='train_g', current_class=None) -> torch.Tensor:
        if train_mode == 'train_f':
            
            f = self.pool_layer(inputs)
            return torch.squeeze(self.head[current_class](self.bottleneck[current_class](f)),dim=1) # b*1
        else:
            f = self.pool_layer(inputs) # b*1024
            output_list = []
            for i in range(self.num_classes):
                output_list.append(torch.squeeze(self.head[i](self.bottleneck[i](f)))) # b*1
            output = torch.stack(output_list,dim=1) # b*6
            return output
        



class ImageClassifierHeadMode3(nn.Module):
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


    def __init__(self, in_features: int, num_classes: int, bottleneck_dim: Optional[int] = 1024, pool_layer=None):
        super(ImageClassifierHeadMode3, self).__init__()
        self.num_classes = num_classes
        if pool_layer is None:
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        else:
            self.pool_layer = pool_layer
        self.bottleneck = nn.ModuleList(
            [nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
                nn.ReLU()) 
                for i in range(num_classes)]
        )
        self.head  = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(bottleneck_dim, 1)) 
                for i in range(num_classes)]
        )

    def forward(self, inputs: torch.Tensor, train_mode='train_g', current_class=None) -> torch.Tensor:
        if train_mode == 'train_f':
            
            f = self.pool_layer(inputs)
            return torch.squeeze(self.head[current_class](self.bottleneck[current_class](f)),dim=1) # b*1
        else:
            f = self.pool_layer(inputs) # b*1024
            output_list = []
            for i in range(self.num_classes):
                output_list.append(torch.squeeze(self.head[i](self.bottleneck[i](f)))) # b*1
            output = torch.stack(output_list,dim=1) # b*6
            return output
        



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


    def __init__(self, in_features: int, num_classes: int, bottleneck_dim: Optional[int] = 1024, pool_layer=None):
        super(ImageClassifierHeadMode4, self).__init__()
        self.num_classes = num_classes
        if pool_layer is None:
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        else:
            self.pool_layer = pool_layer
        self.bottleneck = nn.Identity()
        self.head  = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(in_features, 1)) 
                for i in range(num_classes)]
        )

    def forward(self, inputs: torch.Tensor, current_class=None) -> torch.Tensor:
        if current_class is not None:
            f = self.pool_layer(inputs)
            return torch.squeeze(self.head[current_class](f),dim=1) # b*1
        else:
            f = self.pool_layer(inputs) # b*1024
            output_list = []
            for i in range(self.num_classes):
                output_list.append(torch.squeeze(self.head[i](f))) # b*1
            output = torch.stack(output_list,dim=1) # b*6
            return output