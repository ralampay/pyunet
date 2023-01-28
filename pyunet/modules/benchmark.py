import sys
import os
import torch
import cv2
from matplotlib import pyplot as plt
import numpy as np
import glob
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.unet import UNet
from lib.unet_rd import UNetRd
from lib.utils import get_image, get_mask, get_predicted_img, dice_score, count_parameters

class Benchmark:
    def __init__(self, params={}):
        self.params = params

        self.img_width              = params.get('img_height')
        self.img_height             = params.get('img_height')
        self.device                 = params.get('device')
        self.gpu_index              = params.get('gpu_index')
        self.input_img_dir          = params.get('input_img_dir')
        self.input_mask_dir         = params.get('input_mask_dir')
        self.model_file             = params.get('model_file')
        self.model_type             = params.get('model_type')
        self.in_channels            = params.get('in_channels') or 3
        self.out_channels           = params.get('out_channels') or 3

    def execute(self):
        print("Mode: Benchmark")
        print("Input Image Dir: {}".format(self.input_img_dir))
        print("Input Mask Dir:{}".format(self.input_mask_dir))
        print("Model Type: {}".format(self.model_type))

        if self.device == 'cuda':
            print("CUDA Device: {}".format(torch.cuda.get_device_name(self.gpu_index)))
            self.device = "cuda:{}".format(self.gpu_index)

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

        state = torch.load(
            self.model_file,
            map_location=self.device
        )

        self.model.load_state_dict(state['state_dict'])
        self.model.optimizer = state['optimizer']

        test_images = sorted(glob.glob("{}/*".format(self.input_img_dir)))
        test_masks  = sorted(glob.glob("{}/*".format(self.input_mask_dir)))

        dim = (self.img_width, self.img_height)

        num_images = len(test_images)

        ave_accuracy    = 0.0
        ave_f1          = 0.0
        ave_precision   = 0.0
        ave_recall      = 0.0
        ave_specificity = 0.0
        ave_jaccard     = 0.0

        for i in range(num_images):
            image_file  = test_images[i]
            mask_file   = test_masks[i]

            img  = get_image(image_file, dim)
            mask = get_mask(mask_file, dim)

            prediction = get_predicted_img(img, self.model, device=self.device)

            mask_vectorized = mask.ravel().astype(int)
            prediction_vectorized = prediction.ravel().astype(int)

            accuracy    = accuracy_score(mask_vectorized, prediction_vectorized)
            f1          = f1_score(mask_vectorized, prediction_vectorized, average='macro', zero_division=1)
            precision   = precision_score(mask_vectorized, prediction_vectorized, average='macro', zero_division=1)
            recall      = recall_score(mask_vectorized, prediction_vectorized, average='macro', zero_division=1) # sensitivity
            specificity = recall_score(mask_vectorized, prediction_vectorized, labels=range(self.out_channels), average='macro', zero_division=1)
            jaccard     = jaccard_score(mask_vectorized, prediction_vectorized, labels=range(self.out_channels), average='macro')

            ave_accuracy += accuracy
            ave_f1 += f1
            ave_precision += precision
            ave_recall += recall
            ave_specificity += specificity
            ave_jaccard += jaccard

        ave_accuracy    = ave_accuracy / num_images
        ave_f1          = ave_f1 / num_images
        ave_precision   = ave_precision / num_images
        ave_recall      = ave_recall / num_images
        ave_specificity = ave_specificity / num_images
        ave_jaccard     = ave_jaccard / num_images

        scores = [
            {
                'model_type':       self.model_type,
                'num_params':       count_parameters(self.model),
                'ave_accuary':      ave_accuracy,
                'ave_f1':           ave_f1,
                'ave_precision':    ave_precision,
                'ave_recall':       ave_recall,
                'ave_specificity':  ave_specificity,
                'ave_jaccard':      ave_jaccard
            }
        ]

        df_results = pd.DataFrame(scores)
        print(df_results)
