import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DepthwiseSeperableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2):
        super(DepthwiseSeperableConv, self).__init__()

        self.depthwise      = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False, dilation=dilation)
        self.bn_depthwise   = nn.BatchNorm2d(in_channels)
        self.relu_depthwise = nn.SELU(inplace=True)

        self.pointwise      = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_pointwise   = nn.BatchNorm2d(out_channels)
        self.relu_pointwise = nn.SELU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)
            self.skip_bn = nn.BatchNorm2d(out_channels)
        else:
            self.skip_conv = None

    def forward(self, x):
        identity = x

        out = self.depthwise(x)
        out = self.bn_depthwise(out)
        out = self.relu_depthwise(out)

        out = self.pointwise(out)
        #out = self.bn_pointwise(out)

        if self.skip_conv is not None:
            identity = self.skip_conv(identity)
            identity = self.skip_bn(identity)

        out += identity
        out = self.relu_pointwise(out)

        return out
