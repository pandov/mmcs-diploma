import torch

from torch import Tensor
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

        with torch.set_grad_enabled(self.is_train_loader):
            outputs = self.model(inputs)
            loss = self._calc_loss(outputs, targets)
            if self.is_train_loader:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            outputs = outputs.detach()
            self.batch_metrics.update({
                'loss': loss,
                'lr': self.scheduler.get_last_lr()[0],
                **self._calc_metrics(outputs, targets),
            })

    def on_epoch_end(self, runner):
        super().on_epoch_end(runner)
        self.scheduler.step()
