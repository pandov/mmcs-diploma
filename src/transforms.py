from typing import Tuple, Union
from torch import Tensor
from PIL.Image import Image

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


class Compose(T.Compose):
    def __call__(self, image: Union[Tensor, Image], mask: Union[Tensor, Image]) -> Tuple[Union[Tensor, Image]]:
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class SingleChannel(object):
    def __call__(self, image: Image, mask: Image) -> Tuple[Image]:
        image = image.convert('L')
        mask = mask.convert('1')
        return image, mask

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def forward(self, image: Image, mask: Image) -> Tuple[Image]:
        if torch.rand(1) < self.p:
            image = F.hflip(image)
            mask = F.hflip(mask)
        return image, mask


class RandomVerticalFlip(T.RandomVerticalFlip):
    def forward(self, image: Image, mask: Image) -> Tuple[Image]:
        if torch.rand(1) < self.p:
            image = F.vflip(image)
            mask = F.vflip(mask)
        return image, mask


class FiveCrop(T.FiveCrop):
    def forward(self, image: Image, mask: Image) -> Tuple[Tuple[Image]]:
        images = F.five_crop(image, self.size)
        masks = F.five_crop(mask, self.size)
        return images, masks


class ToTensor(T.ToTensor):
    def __call__(self, images: Tuple[Image], masks: Tuple[Image]) -> Tuple[Tensor]:
        images = torch.stack([F.to_tensor(image) for image in images])
        masks = torch.stack([F.to_tensor(mask) for mask in masks])
        return images, masks
