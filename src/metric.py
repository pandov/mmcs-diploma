from typing import Tuple
from torch import Tensor

import torch
EPS = 1e-7

def get_intersection_union(outputs: Tensor, targets: Tensor) -> Tuple[Tensor]:
    intersection = (outputs * targets).sum(dim=[2, 3])
    union = outputs.sum(dim=[2, 3]) + targets.sum(dim=[2, 3])
    return intersection, union

def dice(outputs: Tensor, targets: Tensor) -> Tensor:
    intersection, union = get_intersection_union(outputs, targets)
    score = (2 * intersection) / (union + EPS)
    return score.mean()

def iou(outputs: Tensor, targets: Tensor) -> Tensor:
    intersection, union = get_intersection_union(outputs, targets)
    score = intersection / (union - intersection + EPS)
    return score.mean()

def accuracy(outputs: Tensor, targets: Tensor, threshold: float = 0.5) -> Tensor:
    score = ((outputs > threshold).float() == targets).float()
    return score.mean()
