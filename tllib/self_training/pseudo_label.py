"""
@author: Baixu Chen
@contact: cbx_99_hasta@outlook.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



def get_mask(hi_threshold: float, lo_threshold: float, y):
    y = y.detach()
    y = torch.sigmoid(y)
    if hi_threshold is not None:
        mask_p = (y > hi_threshold).float()
        mask_n = (y < lo_threshold).float()
        return mask_p, mask_n
    else:
        mask_n = (y < lo_threshold).float()
        return None, mask_n

def get_pseudo_label_acc(hi_threshold: float, lo_threshold: float, y, label_t):

    y = y.detach()
    y = torch.sigmoid(y)
    mask_p = (y > hi_threshold).float()
    acc_p = (label_t[mask_p==1]).sum().mul_(100. / mask_p.sum())
   
    mask_n = (y < lo_threshold).float()
    acc_n = ((1.-label_t)[mask_n==1]).sum().mul_(100. / mask_n.sum())
    return acc_p, acc_n

# 通过预先设定的比例动态筛选样本
def get_hi_confidence_samples_by_proportion(hi_proportion: float, lo_proportion: float, x, y, device, sample_num=None, class_balance=False):
    '''
    返回置信度高的target样本和对应pseudo soft label, 并把这些样本按照 (instance-balance or class-balance) 随机重复直到数量为batch_size
    '''
    y = y.detach()
    score = torch.sigmoid(y)
    thresholded_score = score
    # thresholded_score = torch.where(score > 0.5, torch.tensor(1.).to(device), torch.tensor(0.).to(device))
    indices = torch.nonzero(((score > hi_threshold) | (score < lo_threshold)), as_tuple=False).view(-1)
    indices_p = torch.nonzero((score > hi_threshold), as_tuple=False).view(-1)
    indices_n = torch.nonzero((score < lo_threshold), as_tuple=False).view(-1)
    if indices.numel() == 0:
        return None, None
    if sample_num is not None and sample_num > 0:
        if class_balance and indices_p.numel() > 0 and indices_n.numel() > 0:
            extend_indices_p = indices_p.repeat((int(sample_num/2)) // indices_p.size(0) + 1)
            extend_indices_n = indices_n.repeat(int((sample_num/2)) // indices_n.size(0) + 1)
            extend_indices_p = extend_indices_p[torch.randperm(extend_indices_p.size()[0]).to(device)]
            extend_indices_p = extend_indices_p[:int(sample_num/2)]
            extend_indices_n = extend_indices_n[torch.randperm(extend_indices_n.size()[0]).to(device)]
            extend_indices_n = extend_indices_n[:int(sample_num/2)]
            extend_indices = torch.cat((extend_indices_p, extend_indices_n), dim=0)
            extend_indices = extend_indices[torch.randperm(extend_indices.size()[0]).to(device)]
            return x[extend_indices], thresholded_score[extend_indices]
        else:
            extend_indices = indices.repeat(sample_num // indices.size(0) + 1)
            extend_indices = extend_indices[torch.randperm(extend_indices.size()[0]).to(device)]
            extend_indices = extend_indices[:sample_num]
            return x[extend_indices], thresholded_score[extend_indices]
    else:
        return x[indices], thresholded_score[indices]


def get_hi_confidence_samples(hi_threshold: float, lo_threshold: float, x, y, device, sample_num=None, class_balance=False):
    '''
    返回置信度高的target样本和对应pseudo soft label, 并把这些样本按照 (instance-balance or class-balance) 随机重复直到数量为batch_size
    '''
    y = y.detach()
    score = torch.sigmoid(y)
    thresholded_score = score
    # thresholded_score = torch.where(score > 0.5, torch.tensor(1.).to(device), torch.tensor(0.).to(device))
    indices = torch.nonzero(((score > hi_threshold) | (score < lo_threshold)), as_tuple=False).view(-1)
    indices_p = torch.nonzero((score > hi_threshold), as_tuple=False).view(-1)
    indices_n = torch.nonzero((score < lo_threshold), as_tuple=False).view(-1)
    if indices.numel() == 0:
        return None, None
    if sample_num is not None and sample_num > 0:
        if class_balance and indices_p.numel() > 0 and indices_n.numel() > 0:
            extend_indices_p = indices_p.repeat((int(sample_num/2)) // indices_p.size(0) + 1)
            extend_indices_n = indices_n.repeat(int((sample_num/2)) // indices_n.size(0) + 1)
            extend_indices_p = extend_indices_p[torch.randperm(extend_indices_p.size()[0]).to(device)]
            extend_indices_p = extend_indices_p[:int(sample_num/2)]
            extend_indices_n = extend_indices_n[torch.randperm(extend_indices_n.size()[0]).to(device)]
            extend_indices_n = extend_indices_n[:int(sample_num/2)]
            extend_indices = torch.cat((extend_indices_p, extend_indices_n), dim=0)
            extend_indices = extend_indices[torch.randperm(extend_indices.size()[0]).to(device)]
            return x[extend_indices], thresholded_score[extend_indices]
        else:
            extend_indices = indices.repeat(sample_num // indices.size(0) + 1)
            extend_indices = extend_indices[torch.randperm(extend_indices.size()[0]).to(device)]
            extend_indices = extend_indices[:sample_num]
            return x[extend_indices], thresholded_score[extend_indices]
    else:
        return x[indices], thresholded_score[indices]
    

def get_hi_confidence_samples_with_hard_label(hi_threshold: float, lo_threshold: float, x, y, device, sample_num=None):
    '''
    返回置信度高的target样本和对应pseudo soft label, 并把这些样本按照 (instance-balance or class-balance) 随机重复直到数量为batch_size
    '''
    y = y.detach()
    score = torch.sigmoid(y)
    #thresholded_score = score
    thresholded_score = torch.where(score > 0.5, torch.tensor(1.).to(device), torch.tensor(0.).to(device))
    indices = torch.nonzero(((score > hi_threshold) | (score < lo_threshold)), as_tuple=False).view(-1)
    if indices.numel() == 0:
        return None, None
    if sample_num is not None and sample_num > 0:
        extend_indices = indices.repeat(sample_num // indices.size(0) + 1)
        extend_indices = extend_indices[torch.randperm(extend_indices.size()[0]).to(device)]
        extend_indices = extend_indices[:sample_num]
        return x[extend_indices], thresholded_score[extend_indices]
    else:
        return x[indices], thresholded_score[indices]
    

    
def get_hi_confidence_n_samples(lo_threshold: float, x, y, device, class_balance=False):
    '''
    返回置信度高的target样本和对应pseudo soft label, 并把这些样本按照 (instance-balance or class-balance) 随机重复直到数量为batch_size
    '''
    y = y.detach()
    score = torch.sigmoid(y)
    thresholded_score = score
    # thresholded_score = torch.where(score > 0.5, torch.tensor(1.).to(device), torch.tensor(0.).to(device))
    
    indices_n = torch.nonzero((score < lo_threshold), as_tuple=False).view(-1)
    if indices_n.numel() == 0:
        return None, None
    if sample_num is not None and sample_num > 0:
        if class_balance and indices_p.numel() > 0 and indices_n.numel() > 0:
            extend_indices_p = indices_p.repeat((int(sample_num/2)) // indices_p.size(0) + 1)
            extend_indices_n = indices_n.repeat(int((sample_num/2)) // indices_n.size(0) + 1)
            extend_indices_p = extend_indices_p[torch.randperm(extend_indices_p.size()[0]).to(device)]
            extend_indices_p = extend_indices_p[:int(sample_num/2)]
            extend_indices_n = extend_indices_n[torch.randperm(extend_indices_n.size()[0]).to(device)]
            extend_indices_n = extend_indices_n[:int(sample_num/2)]
            extend_indices = torch.cat((extend_indices_p, extend_indices_n), dim=0)
            extend_indices = extend_indices[torch.randperm(extend_indices.size()[0]).to(device)]
            return x[extend_indices], thresholded_score[extend_indices]
        else:
            extend_indices = indices.repeat(sample_num // indices.size(0) + 1)
            extend_indices = extend_indices[torch.randperm(extend_indices.size()[0]).to(device)]
            extend_indices = extend_indices[:sample_num]
            return x[extend_indices], thresholded_score[extend_indices]
    else:
        return x[indices], thresholded_score[indices]






class ConfidenceBasedSelfTrainingLoss(nn.Module):
    """
    Self training loss that adopts confidence threshold to select reliable pseudo labels from
    `Pseudo-Label : The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks (ICML 2013)
    <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.664.3543&rep=rep1&type=pdf>`_.

    Args:
        threshold (float): Confidence threshold.

    Inputs:
        - y: unnormalized classifier predictions.
        - y_target: unnormalized classifier predictions which will used for generating pseudo labels.

    Returns:
         A tuple, including
            - self_training_loss: self training loss with pseudo labels.
            - mask: binary mask that indicates which samples are retained (whose confidence is above the threshold).
            - pseudo_labels: generated pseudo labels.

    Shape:
        - y, y_target: :math:`(minibatch, C)` where C means the number of classes.
        - self_training_loss: scalar.
        - mask, pseudo_labels :math:`(minibatch, )`.

    """

    def __init__(self, threshold: float):
        super(ConfidenceBasedSelfTrainingLoss, self).__init__()
        self.threshold = threshold

    def forward(self, y, y_target):
        confidence, pseudo_labels = F.softmax(y_target.detach(), dim=1).max(dim=1)
        mask = (confidence > self.threshold).float()
        self_training_loss = (F.cross_entropy(y, pseudo_labels, reduction='none') * mask).mean()

        return self_training_loss, mask, pseudo_labels
    
class ConfidenceBasedSelfTrainingLossForBinaryClassification(nn.Module):
    """
    """

    def __init__(self, hi_threshold: float, lo_threshold: float):
        super(ConfidenceBasedSelfTrainingLossForBinaryClassification, self).__init__()
        self.hi_threshold = hi_threshold
        self.lo_threshold = lo_threshold
        self.loss_f = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, y, y_target,device):
        score = torch.sigmoid(y_target.detach())
        pseudo_labels = torch.where(score > 0.5, torch.tensor(1.).to(device), torch.tensor(0.).to(device))
        mask = ((score > self.hi_threshold) | (score < self.lo_threshold)).float()
        self_training_loss = (self.loss_f(y, pseudo_labels) * mask).mean()

        return self_training_loss, mask, pseudo_labels
