from typing import Literal

import torch
from torch import nn


class ELBOLoss(nn.Module):
    def __init__(self,
                 reduce: Literal['mean', 'sum'] = 'mean'):
        super(ELBOLoss, self).__init__()
        self.reduce = reduce
        self.reconstruction_loss = nn.MSELoss(reduce=self.reduce)
        self.kl_loss = KullbackLeiblerLoss(reduce=self.reduce)

    def forward(self,
                X_reconstructed: torch.Tensor,
                X_origin: torch.Tensor,
                mu: torch.Tensor,
                log_var: torch.Tensor) -> tuple[torch.Tensor,
                                                torch.Tensor,
                                                torch.Tensor]:
        recon_loss = self.reconstruction_loss(X_reconstructed, X_origin)
        kl_loss = self.kl_loss(log_var, mu)
        loss = recon_loss + kl_loss
        return (loss, recon_loss, kl_loss)


class KullbackLeiblerLoss(nn.Module):
    def __init__(self,
                 reduce: Literal['mean', 'sum'] = 'mean'):
        super(KullbackLeiblerLoss, self).__init__()
        self.reduce = reduce

    def forward(self,
                log_var: torch.Tensor,
                mu: torch.Tensor) -> torch.Tensor:
        losses = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1)
        return (torch.mean(losses, dim=0) if self.reduce == 'mean'
                else torch.sum(losses, dim=0))
