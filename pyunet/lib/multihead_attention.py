import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiheadAttention(nn.Module):
    def __init__(self, in_channels, num_heads):
        super(MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.query_conv = nn.Conv2d(in_channels, in_channels // num_heads, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // num_heads, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        queries = self.query_conv(x).view(batch_size, self.num_heads, -1, height * width).permute(0, 1, 3, 2)  # (B, num_heads, HW, C//num_heads)
        keys = self.key_conv(x).view(batch_size, self.num_heads, -1, height * width)  # (B, num_heads, C//num_heads, HW)
        values = self.value_conv(x).view(batch_size, 1, -1, height * width)  # (B, 1, C, HW)
        
        attention_weights = F.softmax(torch.matmul(queries, keys), dim=-1)  # (B, num_heads, HW, HW)
        attention_output = torch.matmul(values, attention_weights.permute(0, 1, 3, 2))  # (B, 1, C, HW)
        
        attention_output = attention_output.view(batch_size, channels, height, width)
        attention_output = self.out_conv(attention_output)
        
        return attention_output
