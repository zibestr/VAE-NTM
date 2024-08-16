from torch import nn, optim
from src.model.loss import ELBOLoss
from torch.utils.data import DataLoader
import torch
import numpy as np


class UnsupervisedTrainer:
    def __init__(self,
                 model: nn.Module,
                 lr: float,
                 data: DataLoader):
        self.model = model
        self.optimizer = optim.Adam(params=self.model.parameters(),
                                    lr=lr)
        self.loss = ELBOLoss(reduce='mean')
        self.data = data

    def __train_iter(self, X: torch.Tensor):
        logits, mu, log_var = self.model(X)
        loss, recon_loss, kl_loss = self.loss(logits, X, mu, log_var)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _train_loop(self) -> float:
        losses = []

        self.model.train()
        for i, inputs in enumerate(self.data):
            losses.append(self.__train_iter(inputs))

            print(f'Train loop, batch {i + 1} ' +
                  f'loss: {losses[i]}')
        return np.mean(losses)

    def train(self,
              max_epoch: int = 5) -> np.ndarray:
        train_losses = []

        for epoch in range(1, max_epoch + 1):
            print(f'Epoch {epoch}\n--------------------------------')
            train_losses.append(self._train_loop())
            print(f'Epoch {epoch} loss: {train_losses[-1]}')

        return np.array(train_losses)
