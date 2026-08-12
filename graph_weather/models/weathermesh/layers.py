"""
Implementation based off the technical report and this repo: https://github.com/Brayden-Zhang/WeatherMesh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvDownBlock(nn.Module):
    """Downsampling convolutional block with residual connection.

    Can handle both 2D and 3D inputs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        is_3d: bool = False,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        groups: int = 1,
        activation: nn.Module = nn.GELU(),
    ):
        """Build the two convolutions and the strided residual projection.

        The first convolution changes the number of channels while keeping the spatial size,
        the second one applies ``stride`` and so performs the downsampling. The residual
        branch is a 1x1 convolution carrying the same stride, so both branches can be summed.

        Args:
            in_channels (int): number of channels of the input tensor.
            out_channels (int): number of channels produced by the block.
            is_3d (bool): if True use ``Conv3d``/``BatchNorm3d``, otherwise their 2D
                counterparts. Defaults to False.
            kernel_size (int): kernel size of both convolutions. Defaults to 3.
            stride (int): stride of the second convolution and of the residual projection,
                i.e. the downsampling factor. A per-axis tuple is also accepted, which is how
                the encoder keeps the pressure-level depth unchanged. Defaults to 2.
            padding (int): padding of both convolutions. Defaults to 1.
            groups (int): number of blocked connections of both convolutions. Defaults to 1.
            activation (nn.Module): activation applied after the first convolution and after
                the residual sum. Defaults to ``nn.GELU()``.
        """
        super().__init__()

        Conv = nn.Conv3d if is_3d else nn.Conv2d
        Norm = nn.BatchNorm3d if is_3d else nn.BatchNorm2d

        self.conv1 = Conv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn1 = Norm(out_channels)
        self.activation1 = activation

        self.conv2 = Conv(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn2 = Norm(out_channels)
        self.activation2 = activation

        # Residual connection with 1x1 conv to match dimensions
        self.downsample = Conv(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn_down = Norm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the two convolutions and add the projected residual.

        Args:
            x (torch.Tensor): input of shape ``[B, in_channels, H, W]``, or
                ``[B, in_channels, D, H, W]`` when the block was built with ``is_3d=True``.

        Returns:
            torch.Tensor: activated output with ``out_channels`` channels and the strided
            spatial dimensions divided by ``stride``.
        """
        identity = self.bn_down(self.downsample(x))

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.activation2(out)

        return out


class ConvUpBlock(nn.Module):
    """
    Upsampling convolutional block with residual connection. same as downBlock but reversed.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        is_3d: bool = False,
        kernel_size: int = 3,
        scale_factor: int = 2,
        padding: int = 1,
        groups: int = 1,
        activation: nn.Module = nn.GELU(),
    ):
        """Build the two convolutions and the residual projection used after upsampling.

        Both convolutions are stride 1: the spatial size is instead grown in ``forward`` by
        interpolation. The first convolution keeps ``in_channels``, the second one maps to
        ``out_channels``, and the residual branch is a 1x1 convolution doing the same mapping.

        Args:
            in_channels (int): number of channels of the input tensor.
            out_channels (int): number of channels produced by the block.
            is_3d (bool): if True use ``Conv3d``/``BatchNorm3d`` and trilinear interpolation,
                otherwise the 2D counterparts and bilinear interpolation. Defaults to False.
            kernel_size (int): kernel size of both convolutions. Defaults to 3.
            scale_factor (int): factor by which the height and width are multiplied in
                ``forward``. Defaults to 2.
            padding (int): padding of both convolutions. Defaults to 1.
            groups (int): number of blocked connections of both convolutions. Defaults to 1.
            activation (nn.Module): activation applied after the first convolution and after
                the residual sum. Defaults to ``nn.GELU()``.
        """
        super().__init__()

        Conv = nn.Conv3d if is_3d else nn.Conv2d
        Norm = nn.BatchNorm3d if is_3d else nn.BatchNorm2d
        self.is_3d = is_3d
        self.scale_factor = scale_factor

        self.conv1 = Conv(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn1 = Norm(in_channels)
        self.activation1 = activation

        self.conv2 = Conv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn2 = Norm(out_channels)
        self.activation2 = activation

        # Residual connection with 1x1 conv to match dimensions
        self.upsample = Conv(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn_up = Norm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Interpolate the input, then apply the convolutions and add the projected residual.

        For 3D inputs only the last two axes are scaled, so the depth axis is left untouched.

        Args:
            x (torch.Tensor): input of shape ``[B, in_channels, H, W]``, or
                ``[B, in_channels, D, H, W]`` when the block was built with ``is_3d=True``.

        Returns:
            torch.Tensor: activated output with ``out_channels`` channels, height and width
            multiplied by ``scale_factor``.
        """
        # Upsample input
        if self.is_3d:
            x = F.interpolate(
                x,
                scale_factor=(1, self.scale_factor, self.scale_factor),
                mode="trilinear",
                align_corners=False,
            )
        else:
            x = F.interpolate(
                x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
            )

        identity = self.bn_up(self.upsample(x))

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.activation2(out)

        return out
