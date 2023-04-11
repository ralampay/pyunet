import sys
import os
import torch
import cv2
from matplotlib import pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.utils import initialize_model

class ExportOnnx:
    def __init__(self, params={}):
        #self.device         = params.get('device') or 'cuda'
        self.device         = 'cpu'
        self.gpu_index      = params.get('gpu_index')
        self.model_file     = params.get('model_file')
        self.img_width      = params.get('img_width')
        self.img_height     = params.get('img_height')
        self.in_channels    = params.get('in_channels') or 3
        self.out_channels   = params.get('out_channels') or 2
        self.model_type     = params.get('model_type') or 'unet'
        self.dim            = (self.img_width, self.img_height)
        self.export_file    = params.get('export_file') or 'model.onnx'

    def execute(self):
        # Load model
        if self.device == 'cuda':
            print("CUDA Device: {}".format(torch.cuda.get_device_name(self.gpu_index)))
            self.device = "cuda:{}".format(self.gpu_index)

        print("Loading model {}...".format(self.model_file))
        state = torch.load(self.model_file, map_location=self.device)

        print("Using model type: {}".format(self.model_type))
        print("In Channels: {}".format(self.in_channels))
        print("Out Channels: {}".format(self.out_channels))

        model = initialize_model(
            self.in_channels,
            self.out_channels,
            self.model_type,
            self.device
        )

        model.load_state_dict(state['state_dict'])

        # Export to ONNX format
        print("Saving file to {}".format(self.export_file))
        torch.onnx.export(
            model,
            torch.randn(1, 3, self.img_width, self.img_height).to(self.device),
            self.export_file,
            export_params=True,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output']
        )
