"""
@author: Mingyang Liu
@contact: mingyangliu1024@gmail.com
"""
import math
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Any, Tuple
import torch.nn as nn
from tllib.utils.my_loss import focalLoss

# 本函数中的p样本定义为标记浓度高的样本，阈值默认为2
def mixup_p_data(x, y, device, pp_alpha=1.0, pn_a=1.0, pn_b=2.0, threshold=2):
    '''Returns mixed inputs, pairs of targets, and lambda'''


    p_target_num = y.sum(dim=1).detach()
    if pp_alpha > 0:
        pp_lam = np.random.beta(pp_alpha, pp_alpha)
    else:
        pp_lam = 1

    # E(pn_lam) < 0.5 
    if pn_a > 0:
        pn_lam = np.random.beta(pn_a, pp_alpha)
    else:
        pn_lam = 1
    
    # p样本的索引
    p_indices = torch.nonzero(p_target_num >= 2, as_tuple=False).squeeze() # 一维
    # n样本的索引
    n_indices = torch.nonzero(p_target_num < 2, as_tuple=False).squeeze()
    # 如果没筛出p样本，就进行正常的样本融合
    if p_indices.ndim==0 or n_indices.ndim==0 :
        print('没有p样本 或 n样本')
        # 因为筛不出样本，所以就前一半是p样本，后一半是n样本
        p_indices = torch.arange(x.size(0) // 2)
        n_indices = torch.arange(x.size(0) // 2, x.size(0)-x.size(0)//2)


    # 先做浓度高的样本之间的融合
    # 打乱顺序的p样本的索引
    p_new_indices = p_indices[torch.randperm(p_indices.size(0)).to(device)] 
    # p_x = x[p_indices, :]
    pp_y_a = y[p_indices]
    pp_y_b = y[p_new_indices]
    mixed_pp_x = pp_lam * x[p_indices, :] + (1-pp_lam) * x[p_new_indices, :]

    # 再做浓度高的样本与浓度低的样本之间的融合 
    # p_extend_indices = torch.randint(0, p_indices.size()[0]+1, n_indices.size()[0])
    p_extend_indices = p_indices.repeat(n_indices.size()[0] // p_indices.size()[0] + 1)
    p_extend_indices = p_extend_indices[torch.randperm(p_extend_indices.size()[0]).to(device)]
    p_extend_indices = p_extend_indices[:n_indices.size()[0]]
    pn_y_a = y[n_indices]
    pn_y_b = y[p_extend_indices]
    mixed_pn_x = pn_lam * x[n_indices, :] + (1-pn_lam) * x[p_extend_indices, :]

    
    return (mixed_pp_x, pp_y_a, pp_y_b, pp_lam, mixed_pn_x, pn_y_a, pn_y_b, pn_lam)

def mixup_s_t_data(x_s, x_t, y_s, y_t, device, a=1.0, b=1.0, lam_t=None, get_x_t=False):
    '''
    Returns mixed inputs, pairs of targets, and lambda
    x_s b*feat_dim
    x_t b*feat_dim
    '''
    if lam_t is None:
        if a > 0 and b > 0:
            lam_t = np.random.beta(a, b)
        else:
            lam_t = 1

    batch_size = x_s.size()[0]
    if x_t.size(0) == batch_size:
        index_t = torch.randperm(batch_size).to(device)
        mixed_x = lam_t * x_t[index_t, :] + (1-lam_t) * x_s 
        y_a, y_b = y_t[index_t], y_s
    else:
        index_s = torch.randperm(batch_size).to(device)[0:x_t.size(0)]
        index_t = torch.randperm(x_t.size(0)).to(device)
        mixed_x = lam_t * x_t[index_t, :] + (1-lam_t) * x_s[index_s, :] 
        y_a, y_b = y_t[index_t], y_s[index_s]
    if get_x_t:
        return (mixed_x, y_a, y_b, lam_t, x_t[index_t, :])
    else:
        return (mixed_x, y_a, y_b, lam_t)

def mixup_data(x, y, device, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return (mixed_x, y_a, y_b, lam)




def mixup_focal_loss(pred, y_a, y_b, lam, gamma):
    return lam * focalLoss(pred, y_a, gamma=gamma) + (1-lam) * focalLoss(pred, y_b, gamma=gamma)

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



class MixUpSourceTarget(nn.Module):
    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., a_b_sum: Optional[float] = 10., auto_step: Optional[bool] = True, sup: Optional[float]=10.):
        super(MixUpSourceTarget, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step
        self.sum = a_b_sum
        self.flag_5 = True
        self.flag_9 = True
        self.coeff = 0
        self.sup = sup

    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor, y_s: torch.Tensor, y_t: torch.tensor, device, get_x_t=False) -> torch.Tensor:
        """"""
        coeff = self.sup * np.float64(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        self.coeff = coeff
        if coeff > 5 and self.flag_5 :
            print('a 已经超过5啦')
            self.flag_5 = False
        elif coeff > 9 and self.flag_9 :
            print('a 已经超过9啦')
            self.flag_9 = False
        if self.auto_step:
            self.step()
        return mixup_s_t_data(x_s,x_t,y_s,y_t,device,a=coeff, b=self.sup-coeff, get_x_t=get_x_t)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1


class MixUpSourceTargetTestSchedule(nn.Module):
    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 15000., a_b_sum: Optional[float] = 10., auto_step: Optional[bool] = True, sup: Optional[float]=10., eta=0.1):
        super(MixUpSourceTargetTestSchedule, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step
        self.sum = a_b_sum
        self.flag_5 = True
        self.flag_9 = True
        self.coeff = 0
        self.sup = sup
        self.eta = eta

    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor, y_s: torch.Tensor, y_t: torch.tensor, device, get_x_t=False) -> torch.Tensor:
        """"""
        p = min(1., 1.0 * self.iter_num / self.max_iters)
        coeff = self.sup * math.pow((p*p + 0.001*0.001), self.eta)
        self.coeff = coeff
        if coeff > 5 and self.flag_5 :
            print('a 已经超过5啦')
            self.flag_5 = False
        elif coeff > 9 and self.flag_9 :
            print('a 已经超过9啦')
            self.flag_9 = False
        if self.auto_step:
            self.step()
        return mixup_s_t_data(x_s,x_t,y_s,y_t,device,a=coeff, b=self.sup-coeff, get_x_t=get_x_t)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1




class MixUpSourceTargetLinear(nn.Module):
    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 8000., a_b_sum: Optional[float] = 10., auto_step: Optional[bool] = True):
        super(MixUpSourceTargetLinear, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step
        self.sum = a_b_sum
        self.flag_5 = True
        self.flag_9 = True
        self.coeff = 0

    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor, y_s: torch.Tensor, y_t: torch.tensor, device) -> torch.Tensor:
        """"""
        coeff = min(1., self.iter_num / self.max_iters)
        self.coeff = coeff
        if coeff > 0.5 and self.flag_5 :
            print('a 已经超过.5啦')
            self.flag_5 = False
        elif coeff > 0.9 and self.flag_9 :
            print('a 已经超过.9啦')
            self.flag_9 = False
        if self.auto_step:
            self.step()
        return mixup_s_t_data(x_s,x_t,y_s,y_t,device,lam_t=coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1


class SVDSigmaLoss(nn.Module):
    r"""
 
    .. math::
        f = U\SigmaV^T

    Inputs:
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - f_t (tensor): feature representations on target domain, :math:`f^t`

    Shape:
        - f_s, f_t: :math:`(N, F)` where F means the dimension of input features.
        - Outputs: scalar.

    """

    def __init__(self, mode='L1'):
        super(SVDSigmaLoss, self).__init__()
        assert mode in ['L1', 'L2']
        self.mode = mode

    def forward(self, f_s, f_t, f_st, lam_st):
        _, s, _ = torch.svd((1-lam_st)*f_s+lam_st*f_t)
        _, s_st, _ = torch.svd(f_st)
       
        if self.mode == 'L1':
            loss = torch.mean(torch.abs(s-s_st))
        elif self.mode == 'L2':
            loss = F.mse_loss(s, s_st)
        return loss
    
class SVDULoss(nn.Module):
    r"""
 
    .. math::
        f = U\SigmaV^T

    Inputs:
        - f_s (tensor): feature representations on source domain, :math:`f^s`
        - f_t (tensor): feature representations on target domain, :math:`f^t`

    Shape:
        - f_s, f_t: :math:`(N, F)` where F means the dimension of input features.
        - Outputs: scalar.

    """

    def __init__(self, mode='L1'):
        super(SVDULoss, self).__init__()
        assert mode in ['L1', 'L2']
        self.mode = mode

    def forward(self, f_s, f_t, f_st, lam_st):
        u, _, _ = torch.svd((1-lam_st)*f_s+lam_st*f_t)
        u_st, _, _ = torch.svd(f_st)
        print(f'u : {u} ')
        print(f'u_st : {u_st}')
        # loss = torch.pow(s[0], 2) + torch.pow(s_st[0], 2)
        if self.mode == 'L1':
            loss = torch.mean(torch.abs(u - u_st))
        elif self.mode == 'L2':
            loss = F.mse_loss(u, u_st)
        return loss
    


def entropy(predictions: torch.Tensor, reduction='mean') -> torch.Tensor:
    predictions = torch.sigmoid(predictions)
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    #H = H.sum(dim=1)
    if reduction == 'mean':
        return H.mean()
    else:
        return H
    
def my_entropy(predictions: torch.Tensor, reduction='mean', mode=0) -> torch.Tensor:
    predictions = torch.sigmoid(predictions)
    epsilon = 1e-5
    H = -predictions * torch.log(predictions + epsilon)
    base_tensor = H * H + epsilon
    if mode == 0 :
        eta = 1 - predictions.detach()
    elif mode == 1:
        eta = 2*(1-predictions.detach())
    result_tensor = torch.pow(base_tensor, eta)
    #print('here')
    if reduction == 'mean':
        return result_tensor.mean()
    else:
        return result_tensor
