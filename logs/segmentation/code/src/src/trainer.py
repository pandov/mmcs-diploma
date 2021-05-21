import torch

from src.dataset import CracksDataset, DataLoader
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from catalyst.dl import Runner
from typing import Dict


class Trainer(Runner):
    def __init__(self,
        input_key: str,
        target_key: str,
        *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.input_key = input_key
        self.target_key = target_key

    def _calc_loss(self,
        outputs: Tensor,
        targets: Tensor,
        ) -> Tensor:

        raise NotImplementedError

    def _calc_metrics(self,
        outputs: Tensor,
        targets: Tensor) -> Dict[str, Tensor]:

        raise NotImplementedError

    def _handle_batch(self, batch: Tensor):
        inputs = batch[self.input_key]
        targets = batch[self.target_key]

        self.model.train(self.is_train_loader)
        with torch.set_grad_enabled(self.is_train_loader):
            outputs = self.model(inputs)
            loss = self._calc_loss(outputs, targets)
            if self.is_train_loader:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            outputs = outputs.detach()
            self.batch_metrics.update({
                'loss': loss.detach(),
                'lr': self.scheduler.get_last_lr()[0],
                **self._calc_metrics(outputs, targets),
            })

    def _get_datasets(self) -> Dict[str, CracksDataset]:
        return {
            'train': CracksDataset('train'),
            'valid': CracksDataset('valid'),
        }

    def _get_loaders(self,
        batch_size: int,
        ) -> Dict[str, DataLoader]:

        datasets = self._get_datasets()
        return {
            'train': datasets['train'].get_loader(
                batch_size=batch_size,
                shuffle=True,
                drop_last=True),
            'valid': datasets['valid'].get_loader(
                batch_size=batch_size),
        }

    def _get_optimizer(self, model: Module) -> Optimizer:
        return torch.optim.Adam(model.parameters(), lr=1e-2)

    def _get_scheduler(self,
        optimizer: Optimizer) -> _LRScheduler:

        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[4, 32, 48], gamma=0.1)

    def train(self, *args, **kwargs):
        batch_size = kwargs.pop('batch_size', 1)
        loaders = self._get_loaders(batch_size)
        model = kwargs.pop('model')
        optimizer = self._get_optimizer(model)
        scheduler = self._get_scheduler(optimizer)
        kwargs['loaders'] = loaders
        kwargs['model'] = model
        kwargs['optimizer'] = optimizer
        kwargs['scheduler'] = scheduler
        super().train(*args, **kwargs)

    def on_epoch_end(self, runner):
        super().on_epoch_end(runner)
        self.scheduler.step()
