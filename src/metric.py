from typing import Tuple
from torch import Tensor

import torch
EPS = 1e-7

def intersection_union(outputs: Tensor, targets: Tensor) -> Tuple[Tensor]:
    intersection = (outputs * targets).sum(dim=[1, 2, 3])
    union = outputs.sum(dim=[1, 2, 3]) + targets.sum(dim=[1, 2, 3])
    return intersection, union

def dice(outputs: Tensor, targets: Tensor) -> Tensor:
    intersection, union = intersection_union(outputs, targets)
    score = (2 * intersection) / (union + EPS)
    return score

def iou(outputs: Tensor, targets: Tensor) -> Tensor:
    intersection, union = intersection_union(outputs, targets)
    score = intersection / (union - intersection + EPS)
    return score

def accuracy(outputs: Tensor, targets: Tensor, threshold: float = 0.5) -> Tensor:
    score = ((outputs > threshold).float() == targets).float()
    return score.mean()
