import torch
import torch.nn as nn
from ghost_conv import GhostConv

class UpConvGhost(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(UpConvGhost, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            GhostConv(ch_in, ch_out)
        )

    def forward(self,x):
        x = self.up(x)
        return x
