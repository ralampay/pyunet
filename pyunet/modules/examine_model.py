import sys
import os
import torch
import cv2
from matplotlib import pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ..lib.unet import UNet
from ..lib.unet_rd import UNetRd
from ..lib.utils import count_parameters

class ExamineModel:
    def __init__(self, params={}):
        self.params = params

        self.in_channels    = params.get('in_channels')
        self.out_channels   = params.get('out_channels')
        self.model_type     = params.get('model_type')
        self.device         = params.get('device')

    def execute(self):
        if self.model_type == 'unet':
            self.model = UNet(
                in_channels=self.in_channels, 
                out_channels=self.out_channels
            ).to(self.device)

        elif self.model_type == 'unet_rd':
            self.model = UNetRd(
                in_channels=self.in_channels, 
                out_channels=self.out_channels
            ).to(self.device)

        print(self.model)

        num_parameters = count_parameters(self.model)
        print("Number of Parameters: {}".format(num_parameters))
