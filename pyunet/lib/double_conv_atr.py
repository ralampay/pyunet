import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
from attention_conv_2d import AttentionConv2d

class DoubleConvAtr(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvAtr, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.skip_conn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.attention = AttentionConv2d(out_channels, out_channels, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_result = self.conv(x)

        # Attention
        x_result = self.attention(x_result)

        # Residual
        x_result += self.skip_conn(x)

        x_result = self.relu(x_result)

        return x_result
