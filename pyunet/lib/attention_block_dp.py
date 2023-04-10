import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from depthwise_seperable_conv import DepthwiseSeperableConv

class AttentionBlockDp(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlockDp, self).__init__()

        self.W_g = nn.Sequential(
            #nn.Conv2d(F_g, F_int, 1, 1, 0, bias=True),
            DepthwiseSeperableConv(F_g, F_int),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            #nn.Conv2d(F_l, F_int, 1, 1, 0, bias=True),
            DepthwiseSeperableConv(F_l, F_int),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            #nn.Conv2d(F_int, 1, 1, 1, 0, bias=True),
            DepthwiseSeperableConv(F_int, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi
