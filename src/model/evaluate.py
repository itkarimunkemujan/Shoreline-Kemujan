"""Loss and metric functions shared by training and inference-side validation."""
from __future__ import annotations

import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logit)
        prob_flat = prob.flatten(1)
        target_flat = target.flatten(1)
        inter = (prob_flat * target_flat).sum(1)
        dice = (2 * inter + 1e-6) / (prob_flat.sum(1) + target_flat.sum(1) + 1e-6)
        dice_loss = 1 - dice.mean()
        bce_loss = self.bce(logit, target)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred, target = pred.flatten(1), target.flatten(1)
    inter = (pred * target).sum(1)
    return (2 * inter + eps) / (pred.sum(1) + target.sum(1) + eps)
