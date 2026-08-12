"""Core components for the Factorized Attention mechanism.

These blocks are based on the principles of Axial Attention: instead of
attending over the whole flattened 2D grid, attention is applied along one
spatial axis at a time, so the cost scales with the length of a single axis.
"""

from einops import rearrange
from torch import einsum, nn


def FeedFoward(dim, multiply=4, dropout=0.0):
    """Build the standard feed-forward block used in transformer architecture.

    Consists of 2 linear layers with GELU activation and dropouts, in between.

    Args:
        dim (int): number of input and output channels of the block.
        multiply (int): factor by which the inner dimension expands dim. Defaults to 4.
        dropout (float): dropout probability applied after the activation and
            after the output projection. Defaults to 0.0.

    Returns:
        nn.Sequential: block mapping a (..., dim) tensor to a (..., dim) tensor.
    """
    inner_dim = int(dim * multiply)
    return nn.Sequential(
        nn.Linear(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim),
        nn.Dropout(dropout),
    )


class AxialAttention(nn.Module):
    """Performs multi-head self-attention on a single axis of a 2D feature map.

    Core building block for Factorized Attention. The attended axis is moved into
    the sequence dimension while the other spatial axis is folded into the batch
    dimension, then the result is rearranged back to the original 2D grid layout.
    """

    def __init__(self, dim, heads, dim_head=64, dropout=0.0):
        """Initialize the query/key/value and output projections.

        Args:
            dim (int): number of channels of the input and output feature map.
            heads (int): number of attention heads.
            dim_head (int): number of channels per attention head, so the inner
                dimension is dim_head * heads. Defaults to 64.
            dropout (float): dropout probability applied to the attention
                weights. Defaults to 0.0.
        """
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, axis):
        """Forward pass for axial attention.

        Args:
            x: Input tensor of shape (batch, height, width, channels)
            axis: Axis to perform attention on (1 for height, 2 for width)

        Returns:
            Tensor of shape (batch, height, width, channels), i.e. the same shape
            as the input.

        Raises:
            ValueError: if axis is neither 1 nor 2.
        """
        b, h, w, d = x.shape

        # rearrange tensor to isolate attention axis as the sequence dim
        if axis == 1:
            x = rearrange(x, "b h w d -> (b w) h d")  # attention along height
        elif axis == 2:
            x = rearrange(x, "b h w d -> (b h) w d")  # attention along width
        else:
            raise ValueError("Axis must be 1 (height) or 2 (width)")

        # project to query, key and value tensors
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), (q, k, v)
        )  # reshape for multi-head attn

        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale  # attention scores
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        # attn to the value tensors
        out = einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)

        # original 2D grid format
        if axis == 1:
            out = rearrange(out, "(b w) h d -> b h w d", w=w)
        elif axis == 2:
            out = rearrange(out, "(b h) w d -> b h w d", h=h)

        return out


class FactorizedAttention(nn.Module):
    """Combines 2 AxialAttention blocks to perform full factorized attention.

    Attention is applied over a 2D feature map first along height then along
    width, each as a residual step on top of its own layer normalization.
    """

    def __init__(self, dim, heads, dim_head=64, dropout=0.0):
        """Initialize the two axial attention blocks and their layer norms.

        Args:
            dim (int): number of channels of the input feature map.
            heads (int): number of attention heads used by each axial block.
            dim_head (int): number of channels per attention head. Defaults to 64.
            dropout (float): dropout probability applied to the attention
                weights. Defaults to 0.0.
        """
        super().__init__()
        self.attn_height = AxialAttention(dim, heads, dim_head, dropout)
        self.attn_width = AxialAttention(dim, heads, dim_head, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        """Attend along the height axis and then along the width axis.

        Args:
            x: Input tensor of shape (batch, height, width, channels)

        Returns:
            Tensor of the same shape as the input.
        """
        x = x + self.attn_height(self.norm1(x), axis=1)
        x = x + self.attn_width(self.norm2(x), axis=2)
        return x


class FactorizedTransformerBlock(nn.Module):
    """Standalone transformer block using Factorized attention.

    Applies a factorized attention residual followed by a feed-forward residual,
    each computed on a layer-normalized copy of the input.
    """

    def __init__(self, dim, heads, dim_head=64, feedforward_multiplier=4, dropout=0.0):
        """Initialize the attention and feed-forward sub-layers.

        Args:
            dim (int): number of channels of the input feature map.
            heads (int): number of attention heads.
            dim_head (int): number of channels per attention head. Defaults to 64.
            feedforward_multiplier (int): factor by which the feed-forward inner
                dimension expands dim. Defaults to 4.
            dropout (float): dropout probability used by the attention and
                feed-forward sub-layers. Defaults to 0.0.
        """
        super().__init__()
        self.attn = FactorizedAttention(dim, heads, dim_head, dropout)
        self.ffn = FeedFoward(dim, feedforward_multiplier, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        """Run factorized attention and the feed-forward block as residuals.

        Args:
            x: Input tensor of shape (batch, height, width, channels)

        Returns:
            Tensor of the same shape as the input.
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
