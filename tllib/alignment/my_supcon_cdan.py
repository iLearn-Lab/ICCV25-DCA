"""
@author: Mingyang Liu
@contact: mingyangliu1024@gmail.com
"""
from typing import Optional
from typing import Tuple, Optional, List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tllib.modules.classifier import Classifier as ClassifierBase
from tllib.utils.metric import binary_accuracy_original, accuracy
from tllib.modules.grl import WarmStartGradientReverseLayer
from tllib.modules.entropy import entropy


__all__ = ['ConditionalDomainAdversarialLoss', 'ImageClassifier', 'D_bottleneck']


class ConditionalDomainAdversarialLoss(nn.Module):
    r"""The Conditional Domain Adversarial Loss used in `Conditional Adversarial Domain Adaptation (NIPS 2018) <https://arxiv.org/abs/1705.10667>`_

    Conditional Domain adversarial loss measures the domain discrepancy through training a domain discriminator in a
    conditional manner. Given domain discriminator :math:`D`, feature representation :math:`f` and
    classifier predictions :math:`g`, the definition of CDAN loss is

    .. math::
        loss(\mathcal{D}_s, \mathcal{D}_t) &= \mathbb{E}_{x_i^s \sim \mathcal{D}_s} \text{log}[D(T(f_i^s, g_i^s))] \\
        &+ \mathbb{E}_{x_j^t \sim \mathcal{D}_t} \text{log}[1-D(T(f_j^t, g_j^t))],\\

    where :math:`T` is a :class:`MultiLinearMap`  or :class:`RandomizedMultiLinearMap` which convert two tensors to a single tensor.

    Args:
        domain_discriminator (torch.nn.Module): A domain discriminator object, which predicts the domains of
          features. Its input shape is (N, F) and output shape is (N, 1)
        entropy_conditioning (bool, optional): If True, use entropy-aware weight to reweight each training example.
          Default: False
        randomized (bool, optional): If True, use `randomized multi linear map`. Else, use `multi linear map`.
          Default: False
        num_classes (int, optional): Number of classes. Default: -1
        features_dim (int, optional): Dimension of input features. Default: -1
        randomized_dim (int, optional): Dimension of features after randomized. Default: 1024
        reduction (str, optional): Specifies the reduction to apply to the output:
          ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
          ``'mean'``: the sum of the output will be divided by the number of
          elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``

    .. note::
        You need to provide `num_classes`, `features_dim` and `randomized_dim` **only when** `randomized`
        is set True.

    Inputs:
        - g_s (tensor): unnormalized classifier predictions on source domain, :math:`g^s`
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - g_t (tensor): unnormalized classifier predictions on target domain, :math:`g^t`
        - f_t (tensor): feature representations on target domain, :math:`f^t`

    Shape:
        - g_s, g_t: :math:`(minibatch, C)` where C means the number of classes.
        - f_s, f_t: :math:`(minibatch, F)` where F means the dimension of input features.
        - Output: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(minibatch, )`.

    Examples::

        >>> from tllib.modules.domain_discriminator import DomainDiscriminator
        >>> from tllib.alignment.cdan import ConditionalDomainAdversarialLoss
        >>> import torch
        >>> num_classes = 2
        >>> feature_dim = 1024
        >>> batch_size = 10
        >>> discriminator = DomainDiscriminator(in_feature=feature_dim * num_classes, hidden_size=1024)
        >>> loss = ConditionalDomainAdversarialLoss(discriminator, reduction='mean')
        >>> # features from source domain and target domain
        >>> f_s, f_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
        >>> # logits output from source domain adn target domain
        >>> g_s, g_t = torch.randn(batch_size, num_classes), torch.randn(batch_size, num_classes)
        >>> output = loss(g_s, f_s, g_t, f_t)
    """

    def __init__(self, domain_discriminator: nn.Module, entropy_conditioning: Optional[bool] = False,
                 randomized: Optional[bool] = False, num_classes: Optional[int] = -1,
                 features_dim: Optional[int] = -1, randomized_dim: Optional[int] = 1024,
                 reduction: Optional[str] = 'mean', sigmoid=True):
        super(ConditionalDomainAdversarialLoss, self).__init__()
        self.domain_discriminator = domain_discriminator
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True)
        self.entropy_conditioning = entropy_conditioning
        self.sigmoid = sigmoid
        self.reduction = reduction

        if randomized:
            assert num_classes > 0 and features_dim > 0 and randomized_dim > 0
            self.map = RandomizedMultiLinearMap(features_dim, num_classes, randomized_dim)
        else:
            self.map = MultiLinearMap()
        self.bce = lambda input, target, weight: F.binary_cross_entropy(input, target, weight,
                                                                        reduction=reduction) if self.entropy_conditioning \
            else F.binary_cross_entropy(input, target, reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, g_s: torch.Tensor, f_s: torch.Tensor, g_t: torch.Tensor, f_t: torch.Tensor, multilabel2multiclass) -> torch.Tensor:
        # 这里的label是软标签，不需要再进行sigmoid
        g = torch.cat((g_s, g_t), dim=0)
        g = g.detach()
        batch_size = g.size(0)
        if f_s.dim() == 2:
            f = torch.cat((f_s, f_t), dim=0) # (b, f_dim)
        else:
            f = torch.cat((f_s, f_t), dim=1)
        # g = F.softmax(g, dim=1).detach()
        if multilabel2multiclass:
            g = self.transform_label(g)


        h = self.grl(self.map(f, g))
        d = self.domain_discriminator(h)

        weight = 1.0 + torch.exp(-entropy(g))
        weight = weight / torch.sum(weight) * batch_size

        if self.sigmoid:
            d_label = torch.cat((
                torch.ones((g_s.size(0), 1)).to(g_s.device),
                torch.zeros((g_t.size(0), 1)).to(g_t.device),
            ))
            self.domain_discriminator_accuracy = binary_accuracy_original(d, d_label)
            if self.entropy_conditioning:
                return F.binary_cross_entropy(d, d_label, weight.view_as(d), reduction=self.reduction)
            else:
                return F.binary_cross_entropy(d, d_label, reduction=self.reduction)
        else:
            d_label = torch.cat((
                torch.ones((g_s.size(0), )).to(g_s.device),
                torch.zeros((g_t.size(0), )).to(g_t.device),
            )).long()
            self.domain_discriminator_accuracy = accuracy(d, d_label)
            if self.entropy_conditioning:
                raise NotImplementedError("entropy_conditioning")
            return F.cross_entropy(d, d_label, reduction=self.reduction)
        
    def transform_label(self, labels):
        new_labels = torch.zeros((labels.shape[0],2**labels.shape[1]-1)).to(labels.device) # b * 63
        # print(new_labels)
        for num in range(0, 2**labels.shape[1]-1):
            binary_string = bin(num+1)[2:].ljust(6,'0')  # [2:]用于去掉前缀'0b'
            ones = [i for i, bit in enumerate(binary_string[::-1]) if bit == '1']
            zeros = [i for i, bit in enumerate(binary_string[::-1]) if bit == '0']
            for i in range(labels.shape[0]):
                new_labels[i][num] = labels[i][ones].prod() * ((torch.tensor(1.)-labels[i])[zeros].prod())
        return F.softmax(new_labels,dim=1)
    

class RandomizedMultiLinearMap(nn.Module):
    """Random multi linear map

    Given two inputs :math:`f` and :math:`g`, the definition is

    .. math::
        T_{\odot}(f,g) = \dfrac{1}{\sqrt{d}} (R_f f) \odot (R_g g),

    where :math:`\odot` is element-wise product, :math:`R_f` and :math:`R_g` are random matrices
    sampled only once and ﬁxed in training.

    Args:
        features_dim (int): dimension of input :math:`f`
        num_classes (int): dimension of input :math:`g`
        output_dim (int, optional): dimension of output tensor. Default: 1024

    Shape:
        - f: (minibatch, features_dim)
        - g: (minibatch, num_classes)
        - Outputs: (minibatch, output_dim)
    """

    def __init__(self, features_dim: int, num_classes: int, output_dim: Optional[int] = 1024):
        super(RandomizedMultiLinearMap, self).__init__()
        self.Rf = torch.randn(features_dim, output_dim)
        self.Rg = torch.randn(num_classes, output_dim)
        self.output_dim = output_dim

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        f = torch.mm(f, self.Rf.to(f.device))
        g = torch.mm(g, self.Rg.to(g.device))
        output = torch.mul(f, g) / np.sqrt(float(self.output_dim))
        return output


class MultiLinearMap(nn.Module):
    """Multi linear map

    Shape:
        - f: (minibatch, F)
        - g: (minibatch, C)
        - Outputs: (minibatch, F * C)
    """

    def __init__(self):
        super(MultiLinearMap, self).__init__()

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        # 如果有多个bottleneck
        if f.dim() == 3:

            batch_size = f.size(1) # (num_classes, b, feature_dim)
            #print(f.shape)
            f = f.permute(1, 0, 2) # (b, num_classes, feature_dim)
            #print(f.shape)
            #print(g.shape)
            #print(torch.diag_embed(g).shape)

            output = torch.bmm(torch.diag_embed(g), f) # (b, n, n) (b, n, f)
        else:
            batch_size = f.size(0)
            output = torch.bmm(g.unsqueeze(2), f.unsqueeze(1))
        return output.view(batch_size, -1) # (b, nf)
    
 


class ImageClassifier(ClassifierBase):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck_dim: Optional[int] = 256, **kwargs):
        bottleneck = nn.Sequential(
            # nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            # nn.Flatten(),
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        super(ImageClassifier, self).__init__(backbone, num_classes, bottleneck, bottleneck_dim, **kwargs)



class D_bottleneck(nn.Module):
    r"""SupCon ResNet.

    Args:
        in_features (int): Dimension of input features
        num_classes (int): Number of classes
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: 128

    Shape:
        - Inputs: :math:`(minibatch, F_in)` where F_in = `in_features`.
        - Output: :math:`(C, minibatch, F_out)` where C = `num_classes`, F_out = `bottleneck_dim`.
    """

    def __init__(self, in_features: int, num_classes: int, bottleneck_dim: Optional[int] = 512, pool_layer=None):
        super(D_bottleneck, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = bottleneck_dim
        if pool_layer is None:
            self.pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        else:
            self.pool_layer = pool_layer
        self.bottleneck = nn.Sequential(
            nn.Linear(in_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        f = self.bottleneck(self.pool_layer(inputs))
        return f
    
    def get_parameters(self) -> List[Dict]:
        return [{"params": self.bottleneck.parameters(), "lr": 1.}]


def entropy(g):
    return (torch.tensor(1.)-torch.sigmoid(g)).prod(dim=1).mean()