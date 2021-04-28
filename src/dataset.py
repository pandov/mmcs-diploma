from typing import Tuple, List, Dict
from torch import Tensor
from PIL.Image import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from pathlib import Path
from src import transforms

class CracksDataset(Dataset):
    transform = {
        'train': transforms.Compose([
            transforms.SingleChannel(),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.FiveCrop(224),
            transforms.ToTensor(),
        ]),
        'valid': transforms.Compose([
            transforms.SingleChannel(),
            transforms.FiveCrop(224),
            transforms.ToTensor(),
        ]),
    }

    def __init__(self, mode: str):
        self.mode = mode
        self.images = tuple(Path(f'dataset/{mode}/images').rglob('*.jpg'))
        self.masks = tuple(Path(f'dataset/{mode}/masks').rglob('*.jpg'))
        self._check_matches()

    def __len__(self):
        return len(self.images)

    def _check_matches(self):
        for image, mask in zip(self.images, self.masks):
            assert image.name == mask.name, f'Does not match: {image.name} != {mask.name}'

    def image(self, index: int) -> Image:
        return default_loader(self.images[index])

    def mask(self, index: int) -> Image:
        return default_loader(self.masks[index])

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        image = self.image(index)
        mask = self.mask(index)
        images, masks = self.transform[self.mode](image, mask)
        cracks = self.is_cracks_exists(masks)
        masks[cracks == 0] = torch.zeros_like(masks[cracks == 0])
        return {
            'images': images,
            'masks': masks,
            'cracks': cracks,
        }

    @staticmethod
    def is_cracks_exists(masks: Tensor, threshold: float = 0.01) -> Tensor:
        h, w = masks.shape[-2:]
        scale = masks.sum(dim=[1, 2, 3])
        return (scale / (h * w) >= threshold).int()

    @staticmethod
    def _collate_fn(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        images, masks, cracks = zip(*(
            (b['images'], b['masks'], b['cracks']) for b in batch
        ))
        return {
            'images': torch.cat(images),
            'masks': torch.cat(masks),
            'cracks': torch.cat(cracks),
        }

    def get_loader(self, **kwargs) -> DataLoader:
        return DataLoader(
            dataset=self,
            collate_fn=self._collate_fn,
            pin_memory=torch.cuda.is_available(),
            **kwargs
        )
