import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DepthwiseSeperableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeperableConv, self).__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out
