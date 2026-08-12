"""
Implementation based off the technical report and this repo: https://github.com/Brayden-Zhang/WeatherMesh
"""

from dataclasses import dataclass
from typing import List

import dacite
import torch
import torch.nn as nn

from graph_weather.models.weathermesh.decoder import WeatherMeshDecoder, WeatherMeshDecoderConfig
from graph_weather.models.weathermesh.encoder import WeatherMeshEncoder, WeatherMeshEncoderConfig
from graph_weather.models.weathermesh.processor import (
    WeatherMeshProcessor,
    WeatherMeshProcessorConfig,
)

"""
Notes on implementation

To make NATTEN work on a sphere, we implement our own circular padding. At the poles, we use
the bump attention behavior from NATTEN. For position encoding of tokens, we use Rotary
Embeddings.

In the default configuration of WeatherMesh 2, the NATTEN window is 5,7,7 in depth, width,
height, corresponding to a physical size of 14 degrees longitude and latitude. WeatherMesh 2
contains two processors: a 6hr and a 1hr processor. Each is 10 NATTEN layers deep.

Training: distributed shampoo: https://github.com/facebookresearch/optimizers/blob/main/distributed_shampoo/README.md

Fork version of pytorch checkpoint library called matepoint to implement offloading to RAM

TODO: Add bump attention and rotary embeddings for the circular padding and position encoding

"""


@dataclass
class WeatherMeshConfig:
    """Serializable set of arguments for a ``WeatherMesh`` model.

    It carries both the configs of the submodules and the hyperparameters ``WeatherMesh``
    falls back to when a submodule is not supplied.

    Attributes:
        encoder (WeatherMeshEncoderConfig): config of the encoder.
        processors (List[WeatherMeshProcessorConfig]): one config per processor.
        decoder (WeatherMeshDecoderConfig): config of the decoder.
        timesteps (List[int]): timestep, in hours, handled by each processor.
        surface_channels (int): number of surface variables.
        pressure_channels (int): number of variables given on pressure levels.
        pressure_levels (int): number of pressure levels.
        latent_dim (int): number of channels of the latent space.
        encoder_num_conv_blocks (int): convolutional blocks per encoder path.
        encoder_num_transformer_layers (int): attention layers of the encoder.
        encoder_hidden_dim (int): base width of the encoder paths.
        decoder_num_conv_blocks (int): convolutional blocks per decoder path.
        decoder_num_transformer_layers (int): attention layers of the decoder.
        decoder_hidden_dim (int): base width of the decoder paths.
        processor_num_layers (int): attention layers of each processor.
        kernel (tuple): neighborhood attention window, as depth, height, width.
        num_heads (int): number of attention heads.
    """

    encoder: WeatherMeshEncoderConfig
    processors: List[WeatherMeshProcessorConfig]
    decoder: WeatherMeshDecoderConfig
    timesteps: List[int]
    surface_channels: int
    pressure_channels: int
    pressure_levels: int
    latent_dim: int
    encoder_num_conv_blocks: int
    encoder_num_transformer_layers: int
    encoder_hidden_dim: int
    decoder_num_conv_blocks: int
    decoder_num_transformer_layers: int
    decoder_hidden_dim: int
    processor_num_layers: int
    kernel: tuple
    num_heads: int

    @staticmethod
    def from_json(json: dict) -> "WeatherMesh":
        """Build a config from a plain dictionary.

        Args:
            json (dict): mapping holding one entry per field of this dataclass.

        Returns:
            The ``WeatherMeshConfig`` that ``dacite`` builds from ``json``.
        """
        return dacite.from_dict(data_class=WeatherMeshConfig, data=json)

    def to_json(self) -> dict:
        """Convert this config back to a plain dictionary.

        Returns:
            dict: the fields of this dataclass, with the submodule configs nested.
        """
        return dacite.asdict(self)


@dataclass
class WeatherMeshOutput:
    """Pair of fields returned by a forward pass of ``WeatherMesh``.

    Attributes:
        surface (torch.Tensor): decoded surface field, ``[B, surface_channels, H, W]``.
        pressure (torch.Tensor): decoded pressure level field,
            ``[B, pressure_channels, D, H, W]``.
    """

    surface: torch.Tensor
    pressure: torch.Tensor


class WeatherMesh(nn.Module):
    """Encoder, processors and decoder assembled into a forecasting model.

    The encoder maps the surface and pressure level inputs to a latent volume, the processors
    are applied to that volume once per requested forecast step, and the decoder maps the
    result back to surface and pressure level fields. Each submodule can either be passed in
    ready-made or be built here from the matching hyperparameters.
    """

    def __init__(
        self,
        encoder: nn.Module | None,
        processors: List[nn.Module] | None,
        decoder: nn.Module | None,
        timesteps: List[int],
        surface_channels: int | None,
        pressure_channels: int | None,
        pressure_levels: int | None,
        latent_dim: int | None,
        encoder_num_conv_blocks: int | None,
        encoder_num_transformer_layers: int | None,
        encoder_hidden_dim: int | None,
        decoder_num_conv_blocks: int | None,
        decoder_num_transformer_layers: int | None,
        decoder_hidden_dim: int | None,
        processor_num_layers: int | None,
        kernel: tuple | None,
        num_heads: int | None,
    ):
        """Take the given submodules, or build the missing ones from the hyperparameters.

        The arguments that may be None are only read when the submodule they configure is not
        supplied; passing all three submodules makes them unused.

        Args:
            encoder (nn.Module | None): encoder to use, or None to build a
                ``WeatherMeshEncoder``.
            processors (List[nn.Module] | None): processors to use, one per entry of
                ``timesteps``, or None to build that many ``WeatherMeshProcessor``.
            decoder (nn.Module | None): decoder to use, or None to build a
                ``WeatherMeshDecoder``.
            timesteps (List[int]): timestep, in hours, handled by each processor. Its length
                sets how many processors the model holds.
            surface_channels (int | None): number of surface variables, used as the encoder
                input and decoder output width.
            pressure_channels (int | None): number of pressure level variables, used as the
                encoder input and decoder output width.
            pressure_levels (int | None): number of pressure levels, passed to the encoder.
            latent_dim (int | None): number of channels of the latent space.
            encoder_num_conv_blocks (int | None): convolutional blocks per encoder path.
            encoder_num_transformer_layers (int | None): attention layers of the encoder.
            encoder_hidden_dim (int | None): base width of the encoder paths.
            decoder_num_conv_blocks (int | None): convolutional blocks per decoder path.
            decoder_num_transformer_layers (int | None): attention layers of the decoder.
            decoder_hidden_dim (int | None): base width of the decoder paths.
            processor_num_layers (int | None): attention layers of each processor.
            kernel (tuple | None): neighborhood attention window shared by the encoder, the
                processors and the decoder, as depth, height, width.
            num_heads (int | None): number of attention heads.

        Raises:
            AssertionError: if ``processors`` is given and its length differs from the length
                of ``timesteps``.
        """
        super().__init__()
        if encoder is not None:
            self.encoder = encoder
        else:
            self.encoder = WeatherMeshEncoder(
                input_channels_2d=surface_channels,
                input_channels_3d=pressure_channels,
                latent_dim=latent_dim,
                n_pressure_levels=pressure_levels,
                num_conv_blocks=encoder_num_conv_blocks,
                hidden_dim=encoder_hidden_dim,
                kernel_size=kernel,
                num_heads=num_heads,
                num_transformer_layers=encoder_num_transformer_layers,
            )
        if processors is not None:
            assert len(processors) == len(
                timesteps
            ), "Number of processors must match number of timesteps"
            self.processors = processors
        else:
            self.processors = [
                WeatherMeshProcessor(
                    latent_dim=latent_dim,
                    n_layers=processor_num_layers,
                    kernel=kernel,
                    num_heads=num_heads,
                )
                for _ in range(len(timesteps))
            ]
        if decoder is not None:
            self.decoder = decoder
        else:
            self.decoder = WeatherMeshDecoder(
                latent_dim=latent_dim,
                output_channels_2d=surface_channels,
                output_channels_3d=pressure_channels,
                n_conv_blocks=decoder_num_conv_blocks,
                hidden_dim=decoder_hidden_dim,
                kernel_size=kernel,
                num_heads=num_heads,
                num_transformer_layers=decoder_num_transformer_layers,
            )
        self.timesteps = timesteps

    def forward(
        self, surface: torch.Tensor, pressure: torch.Tensor, forecast_steps: int
    ) -> WeatherMeshOutput:
        """Encode the inputs, roll the latent state forward, then decode it.

        Every processor is applied once per forecast step, in the order they are held, so one
        step advances the latent state by all the timesteps of the model together.

        Args:
            surface (torch.Tensor): surface input of shape ``[B, surface_channels, H, W]``.
            pressure (torch.Tensor): pressure level input of shape
                ``[B, pressure_channels, D, H, W]``.
            forecast_steps (int): how many times the whole chain of processors is applied.

        Returns:
            WeatherMeshOutput: the decoded surface and pressure level fields.
        """
        # Encode input
        latent = self.encoder(surface, pressure)

        # Apply processors for each forecast step
        for _ in range(forecast_steps):
            for processor in self.processors:
                latent = processor(latent)

        # Decode output
        surface_out, pressure_out = self.decoder(latent)

        return WeatherMeshOutput(surface=surface_out, pressure=pressure_out)
