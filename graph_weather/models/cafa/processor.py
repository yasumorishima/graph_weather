"""Processor refining the CaFA latent grid with factorized attention."""

import torch
import torch.nn as nn
from einops import rearrange

from .factorize import FactorizedTransformerBlock


class CaFAProcessor(nn.Module):
    """Processor module for CaFA.

    Handles latent feature map through multiple layers of self-attention,
    allowing information to propagate across the entire global grid. The layers
    are a stack of FactorizedTransformerBlock, and the channel-first latent map
    is rearranged to channel-last for the blocks and back again on output.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int = 64,
        feedforward_multiplier: int = 4,
        dropout: float = 0.0,
    ):
        """Initialize the stack of factorized transformer blocks.

        Args:
            dim: No. of input channels/ features
            depth: No. of FactorizedTransformerBlocks to stack
            heads: No. of attention heads in each block
            dim_head: Dimension of each attention head
            feedforward_multiplier: Multiplier for the feedforward network dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                FactorizedTransformerBlock(dim, heads, dim_head, feedforward_multiplier, dropout)
                for _ in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the latent feature map through every transformer block.

        Args:
            x: Input tensor of shape (batch, channels, height, width)

        Returns:
            Refined tensor of same shape
        """
        x = rearrange(x, "b c h w -> b h w c")
        for block in self.blocks:
            x = block(x)
        x = rearrange(x, "b h w c -> b c h w")
        return x
