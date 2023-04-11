import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DepthwiseSeperableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=1, padding=1, dilation=2):
        super(DepthwiseSeperableConv, self).__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False, dilation=dilation)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.relu(out)

        return out
