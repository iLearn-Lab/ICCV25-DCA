from typing import Optional, List, Dict, Tuple, Callable
import torch.nn as nn
import torch.nn.functional as F
import torch

from tllib.modules.grl import WarmStartGradientReverseLayer




def classifier_disagreement(predictions: torch.Tensor) -> torch.Tensor:
    predictions = F.softmax(predictions,dim=0)
    avg = (torch.ones_like(predictions)/predictions.size(0)).detach()
    return -torch.mean(torch.abs(avg - predictions))


def multifaceted_miscrepancy(y_s: torch.Tensor, y_s_adv: torch.Tensor, y_t: torch.Tensor, y_t_adv: torch.Tensor):
    y_s_adv = torch.stack()
    source_loss = torch.mean(torch.abs(y_s, y_s_adv))
    return 


def cross_entropy(predictions: torch.Tensor,label: torch.Tensor) -> torch.Tensor:
    cls_loss_function = nn.BCEWithLogitsLoss(reduction="mean")
    cls_loss_list = []
    for i in range(predictions.size(0)):
        cls_loss_list.append(cls_loss_function(predictions[i],label))
    cls_loss_tensor = torch.stack(cls_loss_list,dim=0)
    # sum 比 mean 更合适 吗
    return torch.mean(cls_loss_tensor)


class GeneralModule(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, num_classifiers:int, bottleneck: nn.Module,
                 heads: nn.ModuleList, adv_head: nn.Module, grl: Optional[WarmStartGradientReverseLayer] = None,
                 finetune: Optional[bool] = True):
        super(GeneralModule, self).__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.bottleneck = bottleneck
        self.heads = heads
        self.num_classifiers = num_classifiers
        # for i in range(num_classifiers):
        #     self.heads.append(head)
        self.adv_head = adv_head
        self.finetune = finetune
        self.grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000,
                                                       auto_step=True) if grl is None else grl

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        features = self.backbone(x)
        features = self.bottleneck(features)

        if self.training:
            f_s, f_t = features.chunk(2, dim=0)
            # 每次调用grl都会自动step，因此只能使用一次grl
            f_adv = self.grl_layer(features)
            f_s_adv, f_t_adv = f_adv.chunk(2, dim=0)
            outputs_list = []
            outputt_list = []
            output_adv = self.adv_head(f_adv)
            for i in range(self.num_classifiers):
                outputs_list.append(self.heads[i](f_s))
                outputt_list.append(self.heads[i](f_t_adv))
            outputs = torch.stack(outputs_list,dim=0)
            outputt = torch.stack(outputt_list,dim=0)
            return outputs, outputt, output_adv
        else:
            output_list = []
            for i in range(self.num_classifiers):
                output_list.append(self.heads[i](features))
            output = torch.stack(output_list,dim=0)
            return output

    def step(self):
        """
        Gradually increase :math:`\lambda` in GRL layer.
        """
        self.grl_layer.step()

    def get_parameters(self, base_lr=1) -> List[Dict]:
        """
        Return a parameters list which decides optimization hyper-parameters,
        such as the relative learning rate of each layer.
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else base_lr},
            {"params": self.bottleneck.parameters(), "lr": base_lr},
            {"params": self.adv_head.parameters(), "lr": base_lr}
        ]
        for i in range(self.num_classifiers):
            params.append({"params": self.heads[i].parameters(), "lr": base_lr})
        
        return params


class ImageClassifier(GeneralModule):
    r"""Classifier for MDD.

    Classifier for MDD has one backbone, one bottleneck, while two classifier heads.
    The first classifier head is used for final predictions.
    The adversarial classifier head is only used when calculating MarginDisparityDiscrepancy.


    Args:
        backbone (torch.nn.Module): Any backbone to extract 1-d features from data
        num_classes (int): Number of classes
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: 1024
        width (int, optional): Feature dimension of the classifier head. Default: 1024
        grl (nn.Module): Gradient reverse layer. Will use default parameters if None. Default: None.
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: True

    Inputs:
        - x (tensor): input data

    Outputs:
        - outputs: logits outputs by the main classifier
        - outputs_adv: logits outputs by the adversarial classifier

    Shape:
        - x: :math:`(minibatch, *)`, same shape as the input of the `backbone`.
        - outputs, outputs_adv: :math:`(minibatch, C)`, where C means the number of classes.

    .. note::
        Remember to call function `step()` after function `forward()` **during training phase**! For instance,

            >>> # x is inputs, classifier is an ImageClassifier
            >>> outputs, outputs_adv = classifier(x)
            >>> classifier.step()

    """

    def __init__(self, backbone: nn.Module, num_classes: int, num_classifiers:int,
                 bottleneck_dim: Optional[int] = 1024, width: Optional[int] = 1024,
                 grl: Optional[WarmStartGradientReverseLayer] = None, finetune=True, pool_layer=None):
        grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000,
                                                       auto_step=False) if grl is None else grl

        if pool_layer is None:
            pool_layer = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten()
            )
        bottleneck = nn.Sequential(
            pool_layer,
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        bottleneck[1].weight.data.normal_(0, 0.005)
        bottleneck[1].bias.data.fill_(0.1)

        # The classifier head used for final predictions.
        # head = nn.Sequential(
        #     nn.Linear(bottleneck_dim, width),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(width, num_classes)
        # )

        heads  = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(bottleneck_dim, width),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(width, num_classes)) 
                for i in range(num_classifiers)]
            )
        
        
        # The adversarial classifier head
        adv_head = nn.Sequential(
            nn.Linear(bottleneck_dim, width),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(width, num_classes)
        )
        for dep in range(2):
            # head[dep * 3].weight.data.normal_(0, 0.01)
            # head[dep * 3].bias.data.fill_(0.0)
            adv_head[dep * 3].weight.data.normal_(0, 0.01)
            adv_head[dep * 3].bias.data.fill_(0.0)
        for i in range(num_classifiers):
            for dep in range(2):
                heads[i][dep * 3].weight.data.normal_(0, 0.01)
                heads[i][dep * 3].bias.data.fill_(0.0)
        super(ImageClassifier, self).__init__(backbone, num_classes, num_classifiers, bottleneck,
                                              heads, adv_head, grl_layer, finetune)


class ImageRegressor(GeneralModule):
    r"""Regressor for MDD.

    Regressor for MDD has one backbone, one bottleneck, while two regressor heads.
    The first regressor head is used for final predictions.
    The adversarial regressor head is only used when calculating MarginDisparityDiscrepancy.


    Args:
        backbone (torch.nn.Module): Any backbone to extract 1-d features from data
        num_factors (int): Number of factors
        bottleneck_dim (int, optional): Feature dimension of the bottleneck layer. Default: 1024
        width (int, optional): Feature dimension of the classifier head. Default: 1024
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: True

    Inputs:
        - x (Tensor): input data

    Outputs: (outputs, outputs_adv)
        - outputs: outputs by the main regressor
        - outputs_adv: outputs by the adversarial regressor

    Shape:
        - x: :math:`(minibatch, *)`, same shape as the input of the `backbone`.
        - outputs, outputs_adv: :math:`(minibatch, F)`, where F means the number of factors.

    .. note::
        Remember to call function `step()` after function `forward()` **during training phase**! For instance,

            >>> # x is inputs, regressor is an ImageRegressor
            >>> outputs, outputs_adv = regressor(x)
            >>> regressor.step()

    """

    def __init__(self, backbone: nn.Module, num_factors: int, bottleneck = None, head=None, adv_head=None,
                 bottleneck_dim: Optional[int] = 1024, width: Optional[int] = 1024, finetune=True):
        grl_layer = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=False)
        if bottleneck is None:
            bottleneck = nn.Sequential(
                nn.Conv2d(backbone.out_features, bottleneck_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(bottleneck_dim),
                nn.ReLU(),
            )

        # The regressor head used for final predictions.
        if head is None:
            head = nn.Sequential(
                nn.Conv2d(bottleneck_dim, width, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(width),
                nn.ReLU(),
                nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(width),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten(),
                nn.Linear(width, num_factors),
                nn.Sigmoid()
            )
            for layer in head:
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, 0, 0.01)
                    nn.init.constant_(layer.bias, 0)
        # The adversarial regressor head
        if adv_head is None:
            adv_head = nn.Sequential(
                nn.Conv2d(bottleneck_dim, width, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(width),
                nn.ReLU(),
                nn.Conv2d(width, width, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(width),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten(),
                nn.Linear(width, num_factors),
                nn.Sigmoid()
            )
            for layer in adv_head:
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, 0, 0.01)
                    nn.init.constant_(layer.bias, 0)
        super(ImageRegressor, self).__init__(backbone, num_factors, bottleneck,
                                              head, adv_head, grl_layer, finetune)
        self.num_factors = num_factors
