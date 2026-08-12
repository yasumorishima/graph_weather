"""
Implementation based off the technical report and this repo: https://github.com/Brayden-Zhang/WeatherMesh
"""

from dataclasses import dataclass

import dacite
import torch.nn as nn
from natten import NeighborhoodAttention3D


@dataclass
class WeatherMeshProcessorConfig:
    """Serializable set of arguments for a ``WeatherMeshProcessor``.

    Attributes:
        latent_dim (int): number of channels of the latent space.
        n_layers (int): number of neighborhood attention layers.
        kernel (tuple): neighborhood attention window, as depth, height, width.
        num_heads (int): number of attention heads.
    """

    latent_dim: int
    n_layers: int
    kernel: tuple
    num_heads: int

    @staticmethod
    def from_json(json: dict) -> "WeatherMeshProcessor":
        """Build a config from a plain dictionary.

        Args:
            json (dict): mapping holding one entry per field of this dataclass.

        Returns:
            The ``WeatherMeshProcessorConfig`` that ``dacite`` builds from ``json``.
        """
        return dacite.from_dict(data_class=WeatherMeshProcessorConfig, data=json)

    def to_json(self) -> dict:
        """Convert this config back to a plain dictionary.

        Returns:
            dict: the fields of this dataclass.
        """
        return dacite.asdict(self)


class WeatherMeshProcessor(nn.Module):
    """Stack of neighborhood attention layers advancing the latent state by one timestep.

    The processor keeps the shape of the latent volume, so a model can hold one processor per
    timestep and chain them to roll a forecast forward.
    """

    def __init__(self, latent_dim, n_layers=10, kernel=(5, 7, 7), num_heads=8):
        """Build the stack of neighborhood attention layers.

        Args:
            latent_dim (int): number of channels of the latent volume, used as the embedding
                dimension of every layer.
            n_layers (int): number of attention layers to stack. Defaults to 10.
            kernel (tuple): attention window, as depth, height, width. Defaults to (5, 7, 7).
            num_heads (int): number of attention heads. Defaults to 8.
        """
        super().__init__()

        self.layers = nn.ModuleList(
            [
                NeighborhoodAttention3D(
                    embed_dim=latent_dim,
                    num_heads=num_heads,
                    kernel_size=kernel,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        """Run the latent volume through every attention layer in turn.

        Args:
            x: latent volume of shape ``[B, D, H, W, latent_dim]``, the layout produced by
                ``WeatherMeshEncoder``.

        Returns:
            The processed latent volume, with the same shape as ``x``.
        """
        for layer in self.layers:
            x = layer(x)
        return x
