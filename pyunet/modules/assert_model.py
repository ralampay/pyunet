import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.utils import initialize_model, count_parameters

class AssertModel:
    def __init__(self, params={}):
        self.model_type     = params.get('model_type') or 'unet'
        self.in_channels    = params.get('in_channels') or 3
        self.out_channels   = params.get('out_channels') or 2
        self.img_width      = params.get('img_width') or 28
        self.img_height     = params.get('img_height') or 28
        self.device         = params.get('device') or 'cuda'
        self.gpu_index      = params.get('gpu_index') or 0

    def execute(self):
        print(f"Asserting model {self.model_type}")

        if self.device == 'cuda':
            print("CUDA Device: {}".format(torch.cuda.get_device_name(self.gpu_index)))
            self.device = "cuda:{}".format(self.gpu_index)

        self.model = initialize_model(
            self.in_channels,
            self.out_channels,
            self.model_type,
            self.device
        )
       
        tensors = torch.randn(
            1, 
            self.in_channels, 
            self.img_width, 
            self.img_height
        ).to(self.device)

        print(f"Shape of tensors: {tensors.shape}")
        print(f"Datatype of tensors: {tensors.dtype}")
        print(f"Device tensors is stored on: {tensors.device}")

        result = self.model(tensors)

        print(f"Result Shape: {result.shape}")

        num_parameters = count_parameters(self.model)

        print(f"Number of parameters: {num_parameters}")

        print("Done")
