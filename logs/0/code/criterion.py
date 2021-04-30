from torch import Tensor

import torch
from src.metric import dice, iou


class ReducibleLoss(torch.nn.Module):
    def __init__(self, reduction: str = None):
        super().__init__()
        self.reduction = reduction


class DiceLoss(ReducibleLoss):
    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        return 1 - dice(outputs, targets, self.reduction)


class IoULoss(ReducibleLoss):
    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        return 1 - iou(outputs, targets, self.reduction)


class BCEDiceLoss(ReducibleLoss):
    def __init__(self, bce_weight: float = None, dice_weight: float = None, reduction: str = None):
        super().__init__(reduction)
        if bce_weight is None and dice_weight is None:
            self.bce_weight = 0.5
            self.dice_weight = 0.5
        elif bce_weight is not None and dice_weight is None:
            self.bce_weight = bce_weight
            self.dice_weight = 1 - bce_weight
        elif bce_weight is None and dice_weight is not None:
            self.bce_weight = 1 - dice_weight
            self.dice_weight = dice_weight
        else:
            assert (bce_weight + dice_weight) == 1, 'Bad weight: sum is not equal to 1'
            self.bce_weight = bce_weight
            self.dice_weight = dice_weight
        self.bce_loss = torch.nn.BCELoss()
        self.dice_loss = DiceLoss(reduction=self.reduction)

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        if self.bce_weight == 1:
            return self.bce_loss(outputs, targets)
        elif self.dice_weight == 1:
            return self.dice_loss(outputs, targets)
        else:
            bce = self.bce_loss(outputs, targets) * self.bce_weight
            dice = self.dice_loss(outputs, targets) * self.dice_weight
            return bce + dice
