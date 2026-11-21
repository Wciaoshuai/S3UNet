"""Original projection-before-interpolation operators, with stable weight keys."""

from torch import nn
from torch.nn import functional as F


class ProjectUpsample(nn.Module):
    def __init__(self, conv_op, in_channels, out_channels, kernel_size, scale_factor, bias=False):
        super().__init__()
        self.conv = conv_op(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=bias,
        )
        self.scale_factor = scale_factor
        self.mode = "bilinear" if conv_op == nn.Conv2d else "trilinear"

    def forward(self, x):
        return F.interpolate(
            self.conv(x), scale_factor=self.scale_factor, mode=self.mode, align_corners=False
        )
