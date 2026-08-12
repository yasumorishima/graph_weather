"""
Implementation based off the technical report and this repo: https://github.com/Brayden-Zhang/WeatherMesh
"""

from dataclasses import dataclass

import dacite
import einops
import torch
import torch.nn as nn
from natten import NeighborhoodAttention3D

from graph_weather.models.weathermesh.layers import ConvUpBlock


@dataclass
class WeatherMeshDecoderConfig:
    """Serializable set of arguments for a ``WeatherMeshDecoder``.

    Attributes:
        latent_dim (int): number of channels of the latent space.
        output_channels_2d (int): number of surface variables to produce.
        output_channels_3d (int): number of pressure level variables to produce.
        n_conv_blocks (int): number of convolutional blocks in each path.
        hidden_dim (int): base width of the convolutional paths.
        kernel_size (tuple): neighborhood attention window, as depth, height, width.
        num_heads (int): number of attention heads.
        num_transformer_layers (int): number of neighborhood attention layers.
    """

    latent_dim: int
    output_channels_2d: int
    output_channels_3d: int
    n_conv_blocks: int
    hidden_dim: int
    kernel_size: tuple
    num_heads: int
    num_transformer_layers: int

    @staticmethod
    def from_json(json: dict) -> "WeatherMeshDecoder":
        """Build a config from a plain dictionary.

        Args:
            json (dict): mapping holding one entry per field of this dataclass.

        Returns:
            The ``WeatherMeshDecoderConfig`` that ``dacite`` builds from ``json``.
        """
        return dacite.from_dict(data_class=WeatherMeshDecoderConfig, data=json)

    def to_json(self) -> dict:
        """Convert this config back to a plain dictionary.

        Returns:
            dict: the fields of this dataclass.
        """
        return dacite.asdict(self)


class WeatherMeshDecoder(nn.Module):
    """Decoder turning a latent volume back into surface and pressure level fields.

    Neighborhood attention layers are applied first, then a 1x1 convolution widens the latent
    channels and the volume is split along its depth axis: the last entry carries the surface
    and the others the pressure levels. Each part is upsampled by its own stack of
    ``ConvUpBlock`` until it reaches the requested number of output channels.
    """

    def __init__(
        self,
        latent_dim,
        output_channels_2d,
        output_channels_3d,
        n_conv_blocks=3,
        hidden_dim=256,
        kernel_size: tuple = (5, 7, 7),
        num_heads: int = 8,
        num_transformer_layers: int = 3,
    ):
        """Build the attention layers, the channel split and the two upsampling paths.

        Args:
            latent_dim (int): number of channels of the incoming latent volume.
            output_channels_2d (int): number of channels of the returned surface field.
            output_channels_3d (int): number of channels of the returned pressure field.
            n_conv_blocks (int): number of ``ConvUpBlock`` in each path. Every block doubles
                the height and width and halves the width of the features. Defaults to 3.
            hidden_dim (int): base width of both convolutional paths. Defaults to 256.
            kernel_size (tuple): neighborhood attention window, as depth, height, width.
                Defaults to (5, 7, 7).
            num_heads (int): number of attention heads. Defaults to 8.
            num_transformer_layers (int): number of neighborhood attention layers applied to
                the latent volume. Defaults to 3.
        """
        super().__init__()

        # Transformer layers for initial decoding
        self.transformer_layers = nn.ModuleList(
            [
                NeighborhoodAttention3D(
                    embed_dim=latent_dim, num_heads=num_heads, kernel_size=kernel_size
                )
                for _ in range(num_transformer_layers)
            ]
        )

        # Split into pressure levels and surface paths
        self.split = nn.Conv3d(latent_dim, hidden_dim * (2**n_conv_blocks), kernel_size=1)

        # Pressure levels (3D) path
        self.pressure_path = nn.ModuleList(
            [
                ConvUpBlock(
                    hidden_dim * (2 ** (i + 1)),
                    hidden_dim * (2**i) if i > 0 else output_channels_3d,
                    is_3d=True,
                )
                for i in reversed(range(n_conv_blocks))
            ]
        )

        # Surface (2D) path
        self.surface_path = nn.ModuleList(
            [
                ConvUpBlock(
                    hidden_dim * (2 ** (i + 1)),
                    hidden_dim * (2**i) if i > 0 else output_channels_2d,
                )
                for i in reversed(range(n_conv_blocks))
            ]
        )

    def forward(self, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode a latent volume into a surface field and a pressure level field.

        Args:
            latent (torch.Tensor): latent volume of shape ``[B, D, H, W, latent_dim]``, where
                the last depth entry holds the surface, as produced by ``WeatherMeshEncoder``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: the surface field, of shape
            ``[B, output_channels_2d, H', W']``, and the pressure level field, of shape
            ``[B, output_channels_3d, D - 1, H', W']``, with the height and width doubled
            once per convolutional block.
        """
        # Needs to be (B,D,H,W,C) with Batch, Depth (vertical levels), Height, Width, Channels
        # Apply transformer layers
        for transformer in self.transformer_layers:
            latent = transformer(latent)

        latent = einops.rearrange(latent, "B D H W C -> B C D H W")
        # Split features
        features = self.split(latent)
        pressure_features = features[:, :, :-1]
        surface_features = features[:, :, -1:]
        # Decode pressure levels
        for block in self.pressure_path:
            pressure_features = block(pressure_features)
        # Decode surface features
        surface_features = surface_features.squeeze(2)
        for block in self.surface_path:
            surface_features = block(surface_features)

        return surface_features, pressure_features
