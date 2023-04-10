import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from attention_conv_2d import AttentionConv2d
from depthwise_seperable_conv import DepthwiseSeperableConv
from spatial_sse import SpatialSSE

class NormalizedDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1):
        super(NormalizedDoubleConv, self).__init__()

        self.conv = DepthwiseSeperableConv(in_channels, out_channels)
        self.norm = nn.BatchNorm2d(out_channels)
        self.actv = nn.ReLU(inplace=True)

        self.attention = AttentionConv2d(out_channels, out_channels, 1)

        self.skip_conn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # First convolution
        x0 = self.conv(x)
        x0 = self.norm(x0)
        x0 = self.actv(x0)

        x0 = self.attention(x0)

        # Residual
        x0 += self.skip_conn(x)

        return x0
