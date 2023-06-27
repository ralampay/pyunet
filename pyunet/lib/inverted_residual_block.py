import torch
import torch.nn as nn

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=3, stride=1):
        super(InvertedResidualBlock, self).__init__()
        
        self.expanded_channels = in_channels * expansion_factor
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        if expansion_factor != 1:
            layers.append(nn.Conv2d(in_channels, self.expanded_channels, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(self.expanded_channels))
            layers.append(nn.ReLU6(inplace=True))
        
        layers.append(nn.Conv2d(self.expanded_channels, self.expanded_channels, kernel_size=3, stride=stride, padding=1, groups=self.expanded_channels, bias=False))
        layers.append(nn.BatchNorm2d(self.expanded_channels))
        layers.append(nn.ReLU6(inplace=True))
        
        layers.append(nn.Conv2d(self.expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        self.convolution = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.use_residual:
            return x + self.convolution(x)
        else:
            return self.convolution(x)
