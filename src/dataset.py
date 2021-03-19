import torch
from torch.utils.data import Dataset
from pathlib import Path

class CrackSemgentation(Dataset):
    def __init__(self, mode: str):
        self.files = Path(f'dataset/{mode}').rglob('*.jpg')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        pass
