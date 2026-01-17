"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required')

        # 展开 features 为 [batch_size, n_views, feature_dim]
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        # 如果 labels 和 mask 都没有提供，则生成一个自定义的 mask
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size * features.shape[1], dtype=torch.float32).to(device)  # 修正mask的形状
        elif labels is not None:
            labels = labels.contiguous().view(-1)  # 展平为 [2048]
            if labels.shape[0] != batch_size * features.shape[1]:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(device)  # 构建 mask
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        # 展开所有视图为 [batch_size * n_views, feature_dim]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [2048, 128]

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]  # 选择每个样本的第一个视图 [batch_size, feature_dim]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # 计算 logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)

        # 数值稳定性处理
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 扩展 mask 与 logits 匹配
        mask = mask  # 保持 mask 形状与 logits 相同

        # 修复 logits_mask 的形状为 [batch_size * n_views, batch_size * n_views]
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)

        # 确保 logits_mask 和 logits 的形状一致
        assert logits_mask.shape == logits.shape, f"logits_mask shape: {logits_mask.shape}, logits shape: {logits.shape}"

        mask = mask * logits_mask

        # 修复 exp_logits 计算
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # 计算正样本对的均值
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # 最终损失
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        return loss



