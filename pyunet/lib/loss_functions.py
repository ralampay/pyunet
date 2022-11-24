import torch
import torch.nn as nn
from torch.nn import functional as F


# Source: https://github.com/gokulprasadthekkel/pytorch-multi-class-focal-loss/blob/master/focal_loss.py
class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

def dice_loss(p, y, eps=1e-7):
    """Computes the Sorensen-Dice loss

    Source: https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py

    Note that PyTorch optimizers minimize a loss.
    In this case we would like to maximize the dice loss so we return the negated dice loss

    Args:
        y: a tensor of shape [B, 1, H, W]
        p: a tensor of shape [B, C, H, W]. Corresponds to the raw output or logits of the model.
        eps: added to the denominator for numerical stability
    """
    num_classes = p.shape[1]

    if num_classes == 1:
        y_1_hot     = torch.eye(num_classes + 1)[y.squeeze(1)]
        y_1_hot     = y_1_hot.permute(0, 3, 1, 2).float()
        y_1_hot_f   = y_1_hot[:, 0:1, :, :]
        y_1_hot_s   = y_1_hot[:, 1:2, :, :]
        y_1_hot     = torch.cat([y_1_hot_s, y_1_hot_f], dim=1)
        pos_prob    = torch.sigmoid(p)
        neg_prob    = 1 - pos_prob
        probas      = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        y_1_hot = torch.eye(num_classes)[y.squeeze(1)]
        y_1_hot = y_1_hot.permute(0, 3, 1, 2).float()
        probas  = F.softmax(p, dim=1)

    y_1_hot         = y_1_hot.type(p.type())
    dims            = (0,) + tuple(range(2, y.ndimension()))
    intersection    = torch.sum(probas * y_1_hot, dims)
    cardinality     = torch.sum(probas + y_1_hot, dims)
    dice_loss       = (2. * intersection / (cardinality + eps)).mean()

    return (1 - dice_loss)

def tversky_loss(p, y, alpha=1, beta=1, eps=1e-7):
    """Computes the Tversky loss [1].
    Args:
        y: a tensor of shape [B, H, W] or [B, 1, H, W].
        p: a tensor of shape [B, C, H, W]. Corresponds to
        the raw output or p of the model.
        alpha: controls the penalty for false positives.
        beta: controls the penalty for false negatives.
        eps: added to the denominator for numerical stability.

    Returns:
        tversky_loss: the Tversky loss.

    Notes:
        alpha = beta = 0.5 => dice coeff
        alpha = beta = 1 => tanimoto coeff
        alpha + beta = 1 => F beta coeff

    References:
    [1]: https://arxiv.org/abs/1706.05721
    """
    num_classes = p.shape[1]

    if num_classes == 1:
        y_1_hot = torch.eye(num_classes + 1)[y.squeeze(1)]
        y_1_hot = y_1_hot.permute(0, 3, 1, 2).float()
        y_1_hot_f = y_1_hot[:, 0:1, :, :]
        y_1_hot_s = y_1_hot[:, 1:2, :, :]
        y_1_hot = torch.cat([y_1_hot_s, y_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(p)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        y_1_hot = torch.eye(num_classes)[y.squeeze(1)]
        y_1_hot = y_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(p, dim=1)

    y_1_hot = y_1_hot.type(p.type())
    dims = (0,) + tuple(range(2, y.ndimension()))
    intersection = torch.sum(probas * y_1_hot, dims)
    fps = torch.sum(probas * (1 - y_1_hot), dims)
    fns = torch.sum((1 - probas) * y_1_hot, dims)
    num = intersection
    denom = intersection + (alpha * fps) + (beta * fns)
    tversky_loss = (num / (denom + eps)).mean()

    return (1 - tversky_loss)
