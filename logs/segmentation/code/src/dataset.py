from typing import Tuple, List, Dict
from torch import Tensor
from PIL.Image import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from pathlib import Path
from os import cpu_count
from src import augmentations

class CracksDataset(Dataset):
    TRANSFORM = {
        'train': augmentations.Compose([
            augmentations.SingleChannel(),
            augmentations.RandomVerticalFlip(),
            augmentations.RandomHorizontalFlip(),
            augmentations.RandomRotation(),
            augmentations.FiveCrop(224),
            augmentations.ToTensor(),
        ]),
        'valid': augmentations.Compose([
            augmentations.SingleChannel(),
            augmentations.FiveCrop(224),
            augmentations.ToTensor(),
        ]),
    }

    def __init__(self, mode: str):
        self.transform = self.TRANSFORM[mode]
        self.images = list(Path(f'dataset/{mode}/images')\
            .rglob('*.jpg'))
        self.masks = list(Path(f'dataset/{mode}/masks')\
            .rglob('*.jpg'))

    def __len__(self):
        return len(self.images)

    def image(self, index: int) -> Image:
        return default_loader(self.images[index])

    def mask(self, index: int) -> Image:
        return default_loader(self.masks[index])

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        image = self.image(index)
        mask = self.mask(index)
        images, masks = self.transform(image, mask)
        cracks = self.is_cracks_exists(masks)
        masks[cracks == 0] =\
            torch.zeros_like(masks[cracks == 0])
        return {
            'images': images,
            'masks': masks,
            'cracks': cracks,
        }

    @staticmethod
    def is_cracks_exists(
        masks: Tensor,
        threshold: float = 0.001) -> Tensor:

        h, w = masks.shape[-2:]
        scale = masks.sum(dim=[1, 2, 3])
        return (scale / (h * w) >= threshold).int()

    @staticmethod
    def _collate_fn(
        batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:

        images, masks, cracks = zip(*(
            (b['images'], b['masks'], b['cracks'])
            for b in batch
        ))
        return {
            'images': torch.cat(images),
            'masks': torch.cat(masks),
            'cracks': torch.cat(cracks).unsqueeze(1).float(),
        }

    def get_loader(self, **kwargs) -> DataLoader:
        kwargs['num_workers'] = kwargs.pop('num_workers',\
            cpu_count())
        return DataLoader(
            dataset=self,
            collate_fn=self._collate_fn,
            pin_memory=torch.cuda.is_available(),
            **kwargs,
        )
