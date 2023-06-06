import torch
import torch.nn as nn

class SpatialPyramidPooling(nn.Module):
    def __init__(self, output_sizes):
        super(SpatialPyramidPooling, self).__init__()
        self.output_sizes = output_sizes

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        pooled_features = []

        for output_size in self.output_sizes:
            pool = nn.AdaptiveMaxPool2d(output_size)
            pooled = pool(x).view(batch_size, -1)  # Flatten the pooled features
            pooled_features.append(pooled)

        # Concatenate the pooled features along the channel dimension
        x = torch.cat(pooled_features, dim=1)

        return x
