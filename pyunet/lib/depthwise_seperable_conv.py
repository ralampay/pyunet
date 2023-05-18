import torch
import torch.nn as nn

class DepthwiseSeperableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2):
        super(DepthwiseSeperableConv, self).__init__()

        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            groups=in_channels, 
            bias=False, 
            dilation=dilation
        )
        
        self.pointwise = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=False
        )

        self.bn_pointwise = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.bn_pointwise(out)

        out = self.relu(out)

        return out
