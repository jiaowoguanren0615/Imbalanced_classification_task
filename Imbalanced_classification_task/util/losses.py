import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 控制类别平衡的参数
        self.gamma = gamma  # 控制难易样本的调控参数
        self.reduction = reduction

    def forward(self, input, target):
        # 计算交叉熵损失

        target = torch.stack([1 - target, target], dim=1)
        ce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        
        # 计算概率预测
        pt = torch.exp(-ce_loss)
        
        # 计算 Focal Loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # 如果提供了 alpha 参数，应用类别平衡
        if self.alpha is not None:
            alpha = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_loss = alpha * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss