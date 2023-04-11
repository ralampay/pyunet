import sys
import os
import torch
import cv2
from matplotlib import pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.utils import initialize_model

class Forward:
    def __init__(self, params={}):
        self.params = params

        self.device         = params.get('device')
        self.gpu_index      = params.get('gpu_index')
        self.input_img      = params.get('input_img')
        self.model_file     = params.get('model_file')
        self.img_width      = params.get('img_width')
        self.img_height     = params.get('img_height')
        self.model_type     = params.get('model_type') or 'unet'
        self.in_channels    = params.get('in_channels') or 3
        self.out_channels   = params.get('out_channels') or 2
        self.dim            = (self.img_width, self.img_height)

        print("Dimension: {}".format(self.dim))

    def execute(self):
        print("Forwarding for {}...".format(self.input_img))

        if self.device == 'cuda':
            print("CUDA Device: {}".format(torch.cuda.get_device_name(self.gpu_index)))
            self.device = "cuda:{}".format(self.gpu_index)

        print("Loading model {}...".format(self.model_file))
        state = torch.load(self.model_file)

        print("Using model type: {}".format(self.model_type))

        model = initialize_model(self.in_channels, self.out_channels, self.model_type, self.device)

        model.load_state_dict(state['state_dict'])

        # Load image
        img = cv2.imread(self.input_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rows, cols, _ = img.shape
        original_dim = (cols, rows)

        input_image = cv2.resize(img, self.dim) / 255
        input_image = input_image.transpose((2, 0, 1))

        x = torch.Tensor(np.array([input_image])).to(self.device)

        result = model.forward(x)
        result = torch.argmax(result, 1).detach().cpu().numpy().astype(np.float32)
        result = result.transpose((1, 2, 0)) / self.out_channels

        cv2.imshow("result", cv2.resize(result, original_dim))
        cv2.imshow("Original", img)
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 
