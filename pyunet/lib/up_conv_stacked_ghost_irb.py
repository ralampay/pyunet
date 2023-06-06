import torch
import torch.nn as nn
from ghost_conv import GhostConv
from inverted_residual_block import InvertedResidualBlock

class UpConvStackedGhostIrb(nn.Module):
    def __init__(self,ch_in,ch_out,groups=1):
        super(UpConvStackedGhostIrb, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Sequential(
                GhostConv(ch_in, ch_out,groups=groups),
                InvertedResidualBlock(ch_out, ch_out)
            )
        )

    def forward(self,x):
        x = self.up(x)
        return x
