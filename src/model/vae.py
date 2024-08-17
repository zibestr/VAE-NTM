import torch
from torch import nn


class VariationalAutoencoder(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 input_shape: int,
                 device: str = 'auto'):
        super(VariationalAutoencoder, self).__init__()

        if device == 'auto':
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else
                'mps' if torch.backends.mps.is_available() else
                'cpu'
            )
        else:
            self.device = torch.device(device)

        self.encoder = nn.Sequential(
            nn.Linear(in_features=input_shape,
                      out_features=input_shape // 2),
            nn.Linear(in_features=input_shape // 2,
                      out_features=latent_dim)
        ).to(self.device)

        self.latent_space_mu = nn.Linear(latent_dim,
                                         latent_dim).to(self.device)
        self.latent_space_var = nn.Linear(latent_dim,
                                          latent_dim).to(self.device)

        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim,
                      out_features=input_shape // 2),
            nn.Linear(in_features=input_shape // 2,
                      out_features=input_shape),
            nn.Softmax(dim=1)
        ).to(self.device)
        self.is_fitted = False

    def encode(self, X: torch.Tensor) -> tuple[torch.Tensor,
                                               torch.Tensor]:
        encoded = self.encoder(X)
        mu = self.latent_space_mu(encoded)
        log_var = self.latent_space_var(encoded)
        return (mu, log_var)

    def reparametize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        noise = torch.randn_like(std).to(self.device)

        z = mu + noise * std
        return z

    def topic_distribution(self, X: torch.Tensor) -> torch.Tensor:
        # X - vectorized texts
        mu, log_var = self.encode(X)
        z = self.reparametize(mu, log_var)
        theta = nn.functional.softmax(z, dim=1)  # doc-topic distribution
        return theta.detach()

    def sample_words(self,
                     X: torch.Tensor,
                     num_words: int) -> torch.Tensor:
        topic_probs = self.topic_distribution(X)
        words_probs = nn.functional.softmax(
            self.decoder[0].weight.detach() @ topic_probs.T,
            dim=1
        ).T
        return torch.multinomial(
            words_probs,
            num_words
        )

    def topic_distribution_with_words(
        self,
        X: torch.Tensor,
        num_words: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        topic_probs = self.topic_distribution(X)
        words_probs = nn.functional.softmax(
            self.decoder[0].weight.detach() @ topic_probs.T,
            dim=1
        ).T
        values, indices = words_probs.cpu().sort(
            dim=1, descending=True
        )
        return topic_probs, values[:, :num_words], indices[:, :num_words]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        theta = nn.functional.softmax(z, dim=1)
        return self.decoder(theta)

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor,
                                                torch.Tensor,
                                                torch.Tensor]:
        mu, log_var = self.encode(X)
        z = self.reparametize(mu, log_var)
        X_reconst = self.decode(z)
        return (X_reconst, mu, log_var)
