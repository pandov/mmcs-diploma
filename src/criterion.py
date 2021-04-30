from torch import Tensor

import torch
from src.metric import dice, iou


class DiceLoss(torch.nn.Module):
    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        return 1 - dice(outputs, targets)


class IoULoss(torch.nn.Module):
    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        return 1 - iou(outputs, targets)


class BCEDiceLoss(torch.nn.Module):
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.bce_weight = alpha
        self.dice_weight = 1 - alpha
        self.bce_loss = torch.nn.BCELoss()
        self.dice_loss = DiceLoss()

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        bce = self.bce_loss(outputs, targets)
        dice = self.dice_loss(outputs, targets).mean()
        return self.bce_weight * bce + self.dice_weight * dice
