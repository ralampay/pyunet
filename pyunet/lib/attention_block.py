import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class AttentionBlock(nn.Module):
    def __init__(self, g, l, i):
        super(AttentionBlock, self).__init__()

        self.w_g = nn.Sequential(
            nn.Conv2d(g, l, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(g)
        )

        self.w_x = nn.Sequential(
            nn.Conv2d(g, l, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(g)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(i, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.Relu(inplace=True)
    
    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi
        
