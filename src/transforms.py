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


class Transform(object):
    def __repr__(self):
        return self.__class__.__name__ + '()'


class SingleChannel(Transform):
    def __call__(self, image: Image, mask: Image) -> Tuple[Image]:
        # image = image.convert('L')
        mask = mask.convert('1')
        return image, mask


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


class RandomRotation(Transform):
    def __call__(self, image: Image, mask: Image) -> Tuple[Image]:
        angle = 90 * torch.randint(0, 4, (1,)).item()
        image = F.rotate(image, angle)
        mask = F.rotate(mask, angle)
        return image, mask


class Resize(T.Resize):
    def forward(self, image: Image, mask: Image) -> Tuple[Image]:
        image = F.resize(image, self.size, self.interpolation)
        mask = F.resize(mask, self.size, self.interpolation)
        return image, mask


class FiveCrop(T.FiveCrop):
    def forward(self, image: Image, mask: Image) -> Tuple[Tuple[Image]]:
        images = F.five_crop(image, self.size)
        masks = F.five_crop(mask, self.size)
        return images, masks


class ToTensor(T.ToTensor):
    def __call__(self, images: Union[Image, Tuple[Image]], masks: Union[Image, Tuple[Image]]) -> Tuple[Tensor]:
        if isinstance(images, tuple) and isinstance(masks, tuple):
            images = torch.stack([F.to_tensor(image) for image in images])
            masks = torch.stack([F.to_tensor(mask) for mask in masks])
            return images, masks
        elif isinstance(images, Image) and isinstance(masks, Image):
            images = F.to_tensor(images).unsqueeze(0)
            masks = F.to_tensor(masks).unsqueeze(0)
            return images, masks
        else:
            raise NotImplementedError
