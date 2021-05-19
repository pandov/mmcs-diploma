from typing import Tuple
from torch import Tensor

import torch
from torch.nn.functional import binary_cross_entropy as bce
EPS = 1e-5

def intersection_union(outputs: Tensor, targets: Tensor, reduction: str = None) -> Tuple[Tensor]: # dim=[1, 2, 3]
    intersection = (outputs * targets).sum()
    union = outputs.sum() + targets.sum()
    return intersection, union

def dice(outputs: Tensor, targets: Tensor, reduction: str = None) -> Tensor:
    intersection, union = intersection_union(outputs, targets, reduction)
    score = (2 * intersection + EPS) / (union + EPS)
    return score

def iou(outputs: Tensor, targets: Tensor, reduction: str = None) -> Tensor:
    intersection, union = intersection_union(outputs, targets, reduction)
    score = (intersection + EPS) / (union - intersection + EPS)
    return score

def accuracy(outputs: Tensor, targets: Tensor, threshold: float = 0.5) -> Tensor:
    return ((outputs > threshold).int() == targets).float().mean()
