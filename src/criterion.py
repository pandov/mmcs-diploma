from torch import Tensor

import torch
from src.metric import dice, iou


class DiceLoss(torch.nn.Module):
    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        return 1 - dice(outputs, targets)


class IoULoss(torch.nn.Module):
    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        return 1 - iou(outputs, targets)
