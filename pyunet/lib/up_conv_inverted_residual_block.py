import torch
import torch.nn as nn
from inverted_residual_block import InvertedResidualBlock

class UpConvInvertedResidualBlock(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(UpConvInvertedResidualBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            InvertedResidualBlock(ch_in, ch_out)
        )

    def forward(self,x):
        x = self.up(x)
        return x
