import torch.nn as nn
import torch
import torch.nn.functional as F


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        # 把b*n_views*f_dim 变成 (b*n_views)*f_dim
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # print(torch.matmul(anchor_feature, contrast_feature.T)[0,:])
        # print(anchor_dot_contrast[0,:])
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # print(logits[0,:])
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        mask_irrelevant = torch.tensor(1.0) - mask
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        # 原来分母是除了自己的所有正例负例，现改成只有负例
        # 分母是正例负例比较好
        exp_logits = torch.exp(logits) * logits_mask
        # exp_logits = torch.exp(logits) * mask_irrelevant
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)  


        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-6)
        # print(mask.sum(1))
        # print(((mask * log_prob).sum(1)/(mask.sum(1)))[0:10])

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        p_distance = ((anchor_dot_contrast * mask).sum(1) / (mask.sum(1) + 1e-6)).mean()
        n_distance = ((anchor_dot_contrast * mask_irrelevant).sum(1) / (mask_irrelevant.sum(1) + 1e-6)).mean()
        return loss
        #return loss, p_distance, n_distance
    


def entropy(g):
    return (torch.tensor(1.)-torch.sigmoid(g)).prod(dim=1).mean()