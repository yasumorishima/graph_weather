"""
Implementation based off the technical report and this repo: https://github.com/Brayden-Zhang/WeatherMesh
"""

from dataclasses import dataclass

import dacite
import einops
import torch
import torch.nn as nn
from natten import NeighborhoodAttention3D

from graph_weather.models.weathermesh.layers import ConvDownBlock


@dataclass
class WeatherMeshEncoderConfig:
    """Serializable set of arguments for a ``WeatherMeshEncoder``.

    Attributes:
        input_channels_2d (int): number of surface variables.
        input_channels_3d (int): number of variables given on pressure levels.
        latent_dim (int): number of channels of the latent space.
        n_pressure_levels (int): number of pressure levels of the 3D input.
        num_conv_blocks (int): number of convolutional blocks in each path.
        hidden_dim (int): base width of the convolutional paths.
        kernel_size (tuple): neighborhood attention window, as depth, height, width.
        num_heads (int): number of attention heads.
        num_transformer_layers (int): number of neighborhood attention layers.
    """

    input_channels_2d: int
    input_channels_3d: int
    latent_dim: int
    n_pressure_levels: int
    num_conv_blocks: int
    hidden_dim: int
    kernel_size: tuple
    num_heads: int
    num_transformer_layers: int

    @staticmethod
    def from_json(json: dict) -> "WeatherMeshEncoder":
        """Build a config from a plain dictionary.

        Args:
            json (dict): mapping holding one entry per field of this dataclass.

        Returns:
            The ``WeatherMeshEncoderConfig`` that ``dacite`` builds from ``json``.
        """
        return dacite.from_dict(data_class=WeatherMeshEncoderConfig, data=json)

    def to_json(self) -> dict:
        """Convert this config back to a plain dictionary.

        Returns:
            dict: the fields of this dataclass.
        """
        return dacite.asdict(self)


class WeatherMeshEncoder(nn.Module):
    """Encoder mapping surface and pressure level fields to a single latent volume.

    The two inputs are downsampled by separate convolutional paths, a 2D one for the surface
    fields and a 3D one for the pressure level fields, then concatenated along the depth axis
    so that the surface acts as one extra level. A 1x1 convolution projects the result to
    ``latent_dim`` channels and neighborhood attention layers refine it.
    """

    def __init__(
        self,
        input_channels_2d: int,
        input_channels_3d: int,
        latent_dim: int,
        n_pressure_levels: int,
        num_conv_blocks: int = 3,
        hidden_dim: int = 256,
        kernel_size: tuple = (5, 7, 7),
        num_heads: int = 8,
        num_transformer_layers: int = 3,
    ):
        """Build the surface path, the pressure path and the latent projection.

        Args:
            input_channels_2d (int): number of channels of the surface input.
            input_channels_3d (int): number of channels of the pressure level input.
            latent_dim (int): number of channels of the produced latent volume.
            n_pressure_levels (int): number of pressure levels of the 3D input.
            num_conv_blocks (int): number of ``ConvDownBlock`` in each path. Every block
                halves the height and width and doubles the width of the features.
                Defaults to 3.
            hidden_dim (int): base width of both convolutional paths. Defaults to 256.
            kernel_size (tuple): neighborhood attention window, as depth, height, width.
                Defaults to (5, 7, 7).
            num_heads (int): number of attention heads. Defaults to 8.
            num_transformer_layers (int): number of neighborhood attention layers applied to
                the latent volume. Defaults to 3.
        """
        super().__init__()

        # Surface (2D) path
        self.surface_path = nn.ModuleList(
            [
                ConvDownBlock(
                    input_channels_2d if i == 0 else hidden_dim * (2**i),
                    hidden_dim * (2 ** (i + 1)),
                )
                for i in range(num_conv_blocks)
            ]
        )

        # Pressure levels (3D) path
        self.pressure_path = nn.ModuleList(
            [
                ConvDownBlock(
                    input_channels_3d if i == 0 else hidden_dim * (2**i),
                    hidden_dim * (2 ** (i + 1)),
                    stride=(1, 2, 2),  # Want to keep depth the same size
                    is_3d=True,
                )
                for i in range(num_conv_blocks)
            ]
        )

        # Transformer layers for final encoding
        self.transformer_layers = nn.ModuleList(
            [
                NeighborhoodAttention3D(
                    embed_dim=latent_dim, kernel_size=kernel_size, num_heads=num_heads
                )
                for _ in range(num_transformer_layers)
            ]
        )

        # Final projection to latent space
        self.to_latent = nn.Conv3d(hidden_dim * (2**num_conv_blocks), latent_dim, kernel_size=1)

    def forward(self, surface: torch.Tensor, pressure: torch.Tensor) -> torch.Tensor:
        """Encode one surface field and one pressure level field into a latent volume.

        Args:
            surface (torch.Tensor): surface input of shape ``[B, input_channels_2d, H, W]``.
            pressure (torch.Tensor): pressure level input of shape
                ``[B, input_channels_3d, D, H, W]``.

        Returns:
            torch.Tensor: latent volume of shape ``[B, D + 1, H', W', latent_dim]``, where the
            extra depth entry is the encoded surface and ``H'``, ``W'`` are the height and
            width halved once per convolutional block.
        """
        # Process surface data
        for block in self.surface_path:
            surface = block(surface)

        # Process pressure level data
        for block in self.pressure_path:
            pressure = block(pressure)
        # Combine features
        features = torch.cat(
            [pressure, surface.unsqueeze(2)], dim=2
        )  # B C D H W currently, want it to be B D H W C

        # Transform to latent space
        latent = self.to_latent(features)

        # Reshape to get the shapes
        latent = einops.rearrange(latent, "B C D H W -> B D H W C")
        # Apply transformer layers
        for transformer in self.transformer_layers:
            latent = transformer(latent)
        return latent
