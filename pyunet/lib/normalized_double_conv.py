import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class NormalizedDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NormalizedDoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.PReLU(num_parameters=out_channels),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.PReLU(num_parameters=out_channels)
        )

    def forward(self, x):
        return self.conv(x)
