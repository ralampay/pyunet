import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

class AttentionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(AttentionConv2d, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x0 = x
        # calculate attention coefficients
        x = self.conv(x)

        attn_weights = self.softmax(x)
        
        attn_weights = torch.sum(attn_weights, dim=1, keepdim=True)

        x = x0 * attn_weights

        return x
