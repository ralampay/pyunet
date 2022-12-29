import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from depthwise_seperable_conv import DepthwiseSeperableConv

class AttentionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(AttentionConv2d, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.attn = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        # calculate attention coefficients
        x = self.conv(x)
        attn_x = self.attn(x)
        attn_x = self.relu(attn_x)
        #x = torch.mul(x, attn_x)
        x = x + attn_x

        return x

class NormalizedDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1):
        super(NormalizedDoubleConv, self).__init__()

        self.first_conv = DepthwiseSeperableConv(in_channels, out_channels)
        self.first_norm = nn.BatchNorm2d(out_channels)
        self.first_actv = nn.ReLU(inplace=True)

        self.attention = AttentionConv2d(out_channels, out_channels, 1)

        self.skip_conn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # First convolution
        x0 = self.first_conv(x)
        x0 = self.first_norm(x0)
        x0 = self.first_actv(x0)
        #x0 = x0 * x

        # Second convolution
        #x1 = self.second_conv(x0)
        #x1 = self.second_norm(x1)
        #x1 = self.second_actv(x1)
        #x1 = x1 * x0

        x0 = self.attention(x0)

        # Residual
        x0 += self.skip_conn(x)

        return x0
