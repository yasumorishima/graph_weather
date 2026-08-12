"""
Swin 3D Transformer Encoder.

- Uses a 3D convolution for initial feature extraction.
- Applies layer normalization and reshapes data.
- Uses a transformer-based encoder to learn spatial-temporal features.
"""

import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


class Swin3DEncoder(nn.Module):
    """Encodes a 3D volume into a sequence of embeddings.

    A 3D convolution lifts the input channels to the embedding dimension, the result is
    layer normalized and flattened over the depth, height and width axes, and the sequence
    obtained this way is passed through the encoder half of a transformer.
    """

    def __init__(self, in_channels=1, embed_dim=96):
        """Build the convolution, the normalization and the transformer.

        Args:
            in_channels (int): number of channels of the input volume. Defaults to 1.
            embed_dim (int): dimension of the embedding produced for each voxel, and model
                dimension of the transformer. Defaults to 96.
        """
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, embed_dim, kernel_size=3, padding=1, stride=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.swin_transformer = nn.Transformer(
            d_model=embed_dim,
            nhead=8,
            num_encoder_layers=4,
            num_decoder_layers=4,
            dim_feedforward=embed_dim * 4,
        )
        self.embed_dim = embed_dim

        # Define rearrangement patterns using einops
        self.to_transformer_format = Rearrange("b d h w c -> (d h w) b c")
        self.from_transformer_format = Rearrange("(d h w) b c -> b d h w c", d=None, h=None, w=None)

    # To use rearrange function directly instead of the Rearrange layer
    def forward(self, x):
        """Encode a volume into a flat sequence of embeddings.

        The convolution keeps the spatial size, so the sequence has one element per voxel of
        the input volume.

        Args:
            x (torch.Tensor): input volume of shape (batch, in_channels, depth, height,
                width).

        Returns:
            torch.Tensor: sequence of shape (batch, depth * height * width, embed_dim).
        """
        # 3D convolution with einops rearrangement
        x = self.conv1(x)

        # Rearrange for normalization using einops
        x = rearrange(x, "b c d h w -> b d h w c")
        x = self.norm(x)

        # Store spatial dimensions for later reconstruction
        d, h, w = x.shape[1:4]

        # Transform to sequence format for transformer
        x = rearrange(x, "b d h w c -> (d h w) b c")
        x = self.swin_transformer.encoder(x)

        # Restore original spatial structure
        x = rearrange(x, "(d h w) b c -> b (d h w) c", d=d, h=h, w=w)

        # Reshape to the expected output format (batch, seq_len, embed_dim)
        x = rearrange(x, "b (d h w) c -> b (d h w) c", d=d, h=h, w=w)

        return x

    def convolution(self, x):
        """Apply 3D convolution with clear shape transformation."""
        return self.conv1(x)  # b c d h w -> b embed_dim d h w

    def normalization_layer(self, x):
        """Apply layer normalization with einops rearrangement."""
        x = rearrange(x, "b c d h w -> b d h w c")
        return self.norm(x)

    def transformer_encoder(self, x, spatial_dims):
        """
        Apply transformer encoding with proper shape handling.

        Args:
            x (torch.Tensor): Input tensor
            spatial_dims (tuple): Original (depth, height, width) dimensions
        """
        d, h, w = spatial_dims
        x = self.to_transformer_format(x)
        x = self.swin_transformer.encoder(x)
        x = self.from_transformer_format(x, d=d, h=h, w=w)
        return x
