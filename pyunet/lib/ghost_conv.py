import torch
import torch.nn as nn
from depthwise_seperable_conv import DepthwiseSeperableConv
from double_conv import DoubleConv
from attention_conv_2d import AttentionConv2d
from inverted_residual_block import InvertedResidualBlock

class GhostConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, ratio=0.5):
        super(GhostConv, self).__init__()

        self.primary_conv = DoubleConv(
            in_channels,
            int(out_channels * ratio),
            kernel_size,
            stride,
            padding,
            dilation,
            groups=groups,
            bias=False
        )

        self.cheap_conv = DepthwiseSeperableConv(
            in_channels,
            out_channels - int(out_channels * ratio)
        )

        self.bn = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        primary = self.primary_conv(x)
        cheap   = self.cheap_conv(x)

        x = torch.cat([primary, cheap], dim=1)
        x = self.bn(x)
        x = self.relu(x)

        return x
