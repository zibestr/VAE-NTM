import numpy as np
import plotly.express as px
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from src.app.messager import MessageHandler
from src.model.loss import ELBOLoss


class UnsupervisedTrainer:
    def __init__(self,
                 model: nn.Module,
                 lr: float,
                 data: Dataset,
                 batch_size: int,
                 messager: MessageHandler):
        self.model = model
        self.optimizer = optim.Adam(params=self.model.parameters(),
                                    lr=lr)
        self.loss = ELBOLoss(reduce='mean')
        self.data = DataLoader(data, batch_size=batch_size)
        self.messager = messager

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

            self.messager.receive(f'Train loop, batch {i + 1} '
                                  f'loss: {losses[i]}')
        return np.mean(losses)

    def train(self,
              max_epoch: int = 5) -> np.ndarray:
        train_losses = []

        for epoch in range(1, max_epoch + 1):
            self.messager.receive(
                f'Epoch {epoch}\n--------------------------------'
            )
            train_losses.append(self._train_loop())
            self.messager.receive(
                f'Epoch {epoch} loss: {train_losses[-1]}'
            )
        self.model.is_fitted = True

        px.line(x=range(1, max_epoch + 1), y=train_losses,
                title='Train losses', labels={'x': 'Epoch',
                                              'y': 'Loss'}).show()
        return np.array(train_losses)
