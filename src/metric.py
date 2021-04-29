from typing import Tuple
from torch import Tensor

import torch

def intersection_union(outputs: Tensor, targets: Tensor) -> Tuple[Tensor]: # dim=[1, 2, 3]
    intersection = (outputs * targets).sum()
    union = outputs.sum() + targets.sum()
    return intersection, union

def dice(outputs: Tensor, targets: Tensor) -> Tensor:
    intersection, union = intersection_union(outputs, targets)
    score = (2 * intersection + 1) / (union + 1)
    return score

def iou(outputs: Tensor, targets: Tensor) -> Tensor:
    intersection, union = intersection_union(outputs, targets)
    score = (intersection + 1) / (union - intersection + 1)
    return score

def accuracy(outputs: Tensor, targets: Tensor, threshold: float = 0.5) -> Tensor:
    score = ((outputs > threshold).float() == targets).float()
    return score.mean()
