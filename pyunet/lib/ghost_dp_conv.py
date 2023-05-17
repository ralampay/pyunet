import torch
import torch.nn as nn
from depthwise_seperable_conv import DepthwiseSeperableConv

class GhostDpConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2, groups=3, ratio=0.5):
        super(GhostDpConv, self).__init__()

        self.primary_conv = nn.Conv2d(
            in_channels,
            int(out_channels * ratio),
            kernel_size,
            stride,
            padding,
            dilation,
            int(groups * ratio),
            bias=False
        )

        self.cheap_conv = DepthwiseSeperableConv(
            in_channels,
            out_channels - int(out_channels * ratio)
        )

        self.bn = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        primary = self.primary_conv(x)
        cheap   = self.cheap_conv(x)

        x = torch.cat([primary, cheap], dim=1)
        x = self.bn(x)
        x = self.relu(x)

        return x
