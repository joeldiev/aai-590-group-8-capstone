from typing import Iterable, List

import torch
from torch import nn


class DenoisingAutoencoder(nn.Module):
    """
    Generic fully-connected autoencoder for tabular features.

    Architecture is reconstructed from metadata at runtime so the exported
    state_dict can be loaded without hardcoding layer sizes.
    """

    def __init__(
        self,
        input_dim: int,
        encoder_dims: Iterable[int],
        latent_dim: int,
        activation: str = "relu",
        dropout: float = 0.0,
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__()

        encoder_dims = list(encoder_dims)
        if input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be > 0")

        self.input_dim = input_dim
        self.encoder_dims: List[int] = encoder_dims
        self.latent_dim = latent_dim
        self.activation_name = activation
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    def _get_activation(self) -> nn.Module:
        name = self.activation_name.lower()
        if name == "relu":
            return nn.ReLU()
        if name == "elu":
            return nn.ELU()
        if name == "gelu":
            return nn.GELU()
        if name == "leaky_relu":
            return nn.LeakyReLU(0.01)
        raise ValueError(f"Unsupported activation: {self.activation_name}")

    def _dense_block(self, in_dim: int, out_dim: int) -> nn.Sequential:
        layers = [nn.Linear(in_dim, out_dim)]

        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(out_dim))

        layers.append(self._get_activation())

        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))

        return nn.Sequential(*layers)

    def _build_encoder(self) -> nn.Sequential:
        layers = []
        prev_dim = self.input_dim

        for hidden_dim in self.encoder_dims:
            layers.append(self._dense_block(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.latent_dim))
        return nn.Sequential(*layers)

    def _build_decoder(self) -> nn.Sequential:
        layers = []
        prev_dim = self.latent_dim

        for hidden_dim in reversed(self.encoder_dims):
            layers.append(self._dense_block(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, self.input_dim))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction
