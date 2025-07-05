import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""

    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = inputs.permute(
            0, 2, 3, 4, 1).contiguous().view(-1, inputs.size(1))
        targets = targets.view(-1)
        logpt = -F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(logpt)
        focal_loss = -((1 - pt) ** self.gamma) * logpt
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def dice_coefficient(preds, targets, smooth=1):
    """Compute the Dice Coefficient."""
    num_classes = max(preds.max().item(), targets.max().item()) + 1
    preds = F.one_hot(preds, num_classes=num_classes).permute(
        0, 4, 1, 2, 3).float()
    targets = F.one_hot(targets, num_classes=num_classes).permute(
        0, 4, 1, 2, 3).float()
    preds = preds.contiguous().view(-1)
    targets = targets.contiguous().view(-1)
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / \
        (preds.sum() + targets.sum() + smooth)
    return dice.item()


def iou(preds, targets, smooth=1):
    """Compute the Intersection over Union (IoU)."""
    num_classes = max(preds.max().item(), targets.max().item()) + 1
    preds = F.one_hot(preds, num_classes=num_classes).permute(
        0, 4, 1, 2, 3).float()
    targets = F.one_hot(targets, num_classes=num_classes).permute(
        0, 4, 1, 2, 3).float()
    preds = preds.contiguous().view(-1)
    targets = targets.contiguous().view(-1)
    intersection = (preds * targets).sum()
    total = (preds + targets).sum()
    union = total - intersection
    IoU = (intersection + smooth) / (union + smooth)
    return IoU.item()

# To use these in your training script:
# from losses_and_metrics import FocalLoss, dice_coefficient, iou
# criterion = FocalLoss()
