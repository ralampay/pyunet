import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from double_conv import DoubleConv
from attention_block import AttentionBlock
from ghost_conv import GhostConv
from up_conv import UpConv
from up_conv_ghost import UpConvGhost

class UNetAttnGhost(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=1
    ):
        super(UNetAttnGhost, self).__init__()

        alpha = 0.5

        self.in_channels    = in_channels
        self.out_channels   = out_channels

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = GhostConv(in_channels, 64)
        self.conv2 = GhostConv(64, 128)
        self.conv3 = GhostConv(128, 256)
        self.conv4 = GhostConv(256, 512)
        self.conv5 = GhostConv(512, 1024)

        self.up5 = UpConvGhost(1024, 512)
        self.attn5 = AttentionBlock(512, 512, 256)
        self.up_conv5 = GhostConv(1024, 512)

        self.up4 = UpConvGhost(512, 256)
        self.attn4 = AttentionBlock(256, 256, 128)
        self.up_conv4 = GhostConv(512, 256)

        self.up3 = UpConvGhost(256, 128)
        self.attn3 = AttentionBlock(128, 128, 64)
        self.up_conv3 = GhostConv(256, 128)

        self.up2 = UpConvGhost(128, 64)
        self.attn2 = AttentionBlock(64, 64, 32)
        self.up_conv2 = GhostConv(128, 64)

        self.conv_1x1 = GhostConv(64, out_channels)

    def forward(self, x):
        # Encoding path
        x1 = self.conv1(x)

        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)

        x3 = self.maxpool(x2)
        x3 = self.conv3(x3)

        x4 = self.maxpool(x3)
        x4 = self.conv4(x4)

        x5 = self.maxpool(x4)
        x5 = self.conv5(x5)

        # decoding + concat paths
        d5 = self.up5(x5)
        x4 = self.attn5(d5, x4)
        d5 = torch.cat((x4, d5), dim=1)

        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        x3 = self.attn4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)

        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        x2 = self.attn3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)

        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        x1 = self.attn2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)

        d2 = self.up_conv2(d2)

        d1 = self.conv_1x1(d2)

        return d1
