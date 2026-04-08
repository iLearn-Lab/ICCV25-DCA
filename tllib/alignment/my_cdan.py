"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from tllib.modules.grl import WarmStartGradientReverseLayer
from tllib.modules.classifier import Classifier as ClassifierBase
from tllib.utils.metric import binary_accuracy, accuracy
from tllib.modules.entropy import entropy

__all__ = ['ConditionDomainAdversarialLoss']


class ConditionDomainAdversarialLoss(nn.Module):
    r"""
    The Domain Adversarial Loss proposed in
    `Domain-Adversarial Training of Neural Networks (ICML 2015) <https://arxiv.org/abs/1505.07818>`_

    Domain adversarial loss measures the domain discrepancy through training a domain discriminator.
    Given domain discriminator :math:`D`, feature representation :math:`f`, the definition of DANN loss is

    .. math::
        loss(\mathcal{D}_s, \mathcal{D}_t) = \mathbb{E}_{x_i^s \sim \mathcal{D}_s} \text{log}[D(f_i^s)]
            + \mathbb{E}_{x_j^t \sim \mathcal{D}_t} \text{log}[1-D(f_j^t)].

    Args:
        domain_discriminator (torch.nn.Module): A domain discriminator object, which predicts the domains of features. Its input shape is (N, F) and output shape is (N, 1)
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Default: ``'mean'``
        grl (WarmStartGradientReverseLayer, optional): Default: None.

    Inputs:
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - f_t (tensor): feature representations on target domain, :math:`f^t`
        - w_s (tensor, optional): a rescaling weight given to each instance from source domain.
        - w_t (tensor, optional): a rescaling weight given to each instance from target domain.

    Shape:
        - f_s, f_t: :math:`(N, F)` where F means the dimension of input features.
        - Outputs: scalar by default. If :attr:`reduction` is ``'none'``, then :math:`(N, )`.

    Examples::

        >>> from tllib.modules.domain_discriminator import DomainDiscriminator
        >>> discriminator = DomainDiscriminator(in_feature=1024, hidden_size=1024)
        >>> loss = ConditionDomainAdversarialLoss(discriminator, reduction='mean')
        >>> # features from source domain and target domain
        >>> f_s, f_t = torch.randn(20, 1024), torch.randn(20, 1024)
        >>> # If you want to assign different weights to each instance, you should pass in w_s and w_t
        >>> w_s, w_t = torch.randn(20), torch.randn(20)
        >>> output = loss(f_s, f_t, w_s, w_t)
    """

    def __init__(self, domain_discriminator: nn.Module, reduction: Optional[str] = 'mean',
                 grl: Optional = None, sigmoid=True, entropy=False):
        super(ConditionDomainAdversarialLoss, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1., max_iters=1000, auto_step=True) if grl is None else grl
        self.domain_discriminator = domain_discriminator
        self.sigmoid = sigmoid
        self.entropy = entropy
        self.reduction = reduction
        self.bce = lambda input, target, weight: \
            F.binary_cross_entropy(input, target, weight=weight, reduction=reduction)
        self.domain_discriminator_accuracy = None

    def forward(self, h: torch.Tensor, y: torch.Tensor,
                w_s: Optional[torch.Tensor] = None, w_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.grl(h)
        d = self.domain_discriminator(h)
        if self.sigmoid:
            d_s, d_t = d.chunk(2, dim=0)
            y_s, y_t = y.chunk(2, dim=0)
            d_label_s = torch.ones((d_s.size(0), 1)).to(d_s.device)
            d_label_t = torch.zeros((d_t.size(0), 1)).to(d_t.device)
            self.domain_discriminator_accuracy = 0.5 * (binary_accuracy(d_s, d_label_s) + binary_accuracy(d_t, d_label_t))

            if self.entropy:
                # y = F.softmax(y,dim=1).detach()

                w = 1.0 + torch.exp(-entropy(y))
                w = w / (torch.sum(w) / y.size(0))
                w_s, w_t = w.chunk(2, dim=0)

                # w_s = 1.0 + torch.exp(-entropy(y_s))
                # w_s = w_s / torch.sum(w_s) * y_s.size(0)
                #
                # w_t = 1.0 + torch.exp(-entropy(y_t))
                # w_t = w_t / torch.sum(w_t) * y_t.size(0)

                # y_s = F.softmax(y_s, dim=1).detach()
                # y_t = F.softmax(y_t, dim=1).detach()
                # epsilon = 1e-5
                # if w_s is None:
                #     y_s_ylogy = torch.bmm(y_s.view([y_s.size(0), 1, -1]), torch.log(y_s.softmax(dim=-1) + epsilon).view([y_s.size(0), y_s.size(1), -1])).view([y_s.size(0), -1])
                #     w_s = torch.exp(y_s_ylogy) + 1.0
                #     w_s = w_s / torch.sum(w_s) * y_s.size(0)
                #
                # if w_t is None:
                #     y_t_ylogy = torch.bmm(y_t.view([y_t.size(0), 1, -1]),torch.log(y_t.softmax(dim=-1) + epsilon).view([y_t.size(0), y_t.size(1), -1])).view([y_t.size(0), -1])
                #     w_t = torch.exp(y_t_ylogy) + 1.0
                #     w_t = w_t / torch.sum(w_t) * y_t.size(0)
            else:
                if w_s is None:
                    w_s = torch.ones((d_s.size(0),)).to(d_s.device)
                if w_t is None:
                     w_t = torch.ones((d_t.size(0),)).to(d_t.device)

            return 0.5 * (
                F.binary_cross_entropy(d_s, d_label_s, weight=w_s.view_as(d_s), reduction=self.reduction) +
                F.binary_cross_entropy(d_t, d_label_t, weight=w_t.view_as(d_t), reduction=self.reduction)
            )
        else:
            d_s, d_t = d.chunk(2, dim=0)
            d_label = torch.cat((
                torch.ones((d_s.size(0),)).to(d_s.device),
                torch.zeros((d_t.size(0),)).to(d_t.device),
            )).long()
            if w_s is None:
                w_s = torch.ones((d_s.size(0),)).to(d_s.device)
            if w_t is None:
                w_t = torch.ones((d_t.size(0),)).to(d_t.device)
            self.domain_discriminator_accuracy = accuracy(d, d_label)
            loss = F.cross_entropy(d, d_label, reduction='none') * torch.cat([w_s, w_t], dim=0)
            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            elif self.reduction == "none":
                return loss
            else:
                raise NotImplementedError(self.reduction)


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
