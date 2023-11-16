import sys
import os
import torch
import cv2
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import load_model_for_inference

class SamplePair:
    def __init__(self, params={}):
        self.params = params

        self.img_width  = params.get('img_width')
        self.img_height = params.get('img_height')
        
        self.input_img_dir  = params.get('input_img_dir')
        self.input_mask_dir = params.get('input_mask_dir')

        self.model_file = params.get('model_file')
        self.device     = params.get('device') or 'cuda'
        self.gpu_index  = params.get('gpu_index') or 0
        self.model_type = params.get('model_type') or 'unet'
        self.models     = params.get('models') or []

        self.in_channels    = params.get('in_channels')
        self.out_channels   = params.get('out_channels')

        self.sampled_index = params.get('sampled_index') or -1

    def execute(self):
        print("In Channels: {}".format(self.in_channels))
        print("Out Channels: {}".format(self.out_channels))

        print("Sampling pair...")

        print("input_img_dir: {}".format(self.input_img_dir))
        print("input_mask_dir: {}".format(self.input_mask_dir))

        img_list = sorted(os.listdir(self.input_img_dir))
        msk_list = sorted(os.listdir(self.input_mask_dir))

        num_images = len(os.listdir(self.input_img_dir))

        if self.sampled_index < 0:
            self.sampled_index = random.randint(0, num_images - 1)

        print(f"sampled_index: {self.sampled_index}")

        img_path    = os.path.join(self.input_img_dir, img_list[self.sampled_index])
        mask_path   = os.path.join(self.input_mask_dir, msk_list[self.sampled_index])

        print(img_path)
        print(mask_path)

        img_for_plot = cv2.imread(img_path, 1)
        img_for_plot = cv2.cvtColor(img_for_plot, cv2.COLOR_BGR2RGB)

        mask_for_plot = cv2.imread(mask_path, 0)

        dim = (self.img_width, self.img_height)

        img_for_plot    = cv2.resize(img_for_plot, dim)
        mask_for_plot   = cv2.resize(mask_for_plot, dim)

        num_models  = len(self.models)
        num_cols    = num_models + 2
        num_rows    = 1

        subplot_width   = 3
        subplot_height  = 4

        # Create a figure and grid of subplots
        fig, axes = plt.subplots(
            num_rows, 
            num_cols, 
            figsize=(subplot_width * num_cols, subplot_height)
        )

        # Original Image
        axes[0].imshow(img_for_plot)
        axes[0].set_title('Image')
        axes[0].axis('off')

        # Masked Image
        axes[1].imshow(mask_for_plot)
        axes[1].set_title('Mask')
        axes[1].axis('off')

        if self.device == 'cuda':
            print("CUDA Device: {}".format(torch.cuda.get_device_name(self.gpu_index)))
            self.device = "cuda:{}".format(self.gpu_index)

        for model_i, model_cfg in enumerate(self.models):
            model_type = model_cfg['type']

            print(f"Using model type: {model_type}")

            state = torch.load(model_cfg['file'])

            model = load_model_for_inference(
                self.in_channels,
                self.out_channels,
                model_type,
                self.device,
                state['state_dict']
            )

            input_image = cv2.resize(img_for_plot, dim) / 255
            input_image = input_image.transpose((2, 0, 1))

            x = torch.Tensor(np.array([input_image])).to(self.device)

            result = model.forward(x)
            result = torch.argmax(result, 1).detach().cpu().numpy().astype(np.float32)
            result = result.transpose((1, 2, 0)) / self.out_channels

            curr_index = model_i + 2
            axes[curr_index].imshow(result)
            axes[curr_index].set_title(model_type)
            axes[curr_index].axis('off')

        # Adjust the spacing between subplots
        plt.subplots_adjust(wspace=0.2)

        plt.show()
