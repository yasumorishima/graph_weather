"""Encoder mapping the raw weather state onto the CaFA latent grid."""

import torch
from torch import nn


class CaFAEncoder(nn.Module):
    """Encoder for CaFA.

    This projects complex, high-resolution input weather state
    and transform it into a lower-resolution, high-dimensional
    latent representation that the processor can work with.
    The projection is a single strided convolution whose kernel
    size and stride are both the downsampling factor.
    """

    def __init__(self, input_channels: int, model_dim: int, downsampling_factor: int = 1):
        """Initialize the strided convolution used to encode the input grid.

        Args:
            input_channels: No. of channels/features in raw input data
            model_dim: Dimensions of the model's hidden layers (output channels)
            downsampling_factor: Factor to downsample the spatial dimensions by
                (i.e., 2 means H/2, W/2)
        """
        super().__init__()
        self.encoder = nn.Conv2d(
            in_channels=input_channels,
            out_channels=model_dim,
            kernel_size=downsampling_factor,
            stride=downsampling_factor,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode the input weather state into the latent representation.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Encoded tensor of shape (batch, model_dim,
            height/downsampling_factor, width/downsampling_factor)
        """
        x = self.encoder(x)
        return x
