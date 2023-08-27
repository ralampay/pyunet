import sys
import os
import torch
import cv2
from matplotlib import pyplot as plt
import numpy as np
import glob
import time
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.utils import get_image, get_mask, get_predicted_img, dice_score, count_parameters, initialize_model, load_model_for_inference
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
        self.models                 = params.get('models') or []

    def execute(self):
        print("Mode: Benchmark")
        print(f"Input Image Dir: {self.input_img_dir}")
        print(f"Input Mask Dir:{self.input_mask_dir}")
        print(f"In Channels: {self.in_channels}")
        print(f"Out Channels: {self.out_channels}")

        if self.device == 'cuda':
            print(f"CUDA Device: {torch.cuda.get_device_name(self.gpu_index)}")
            self.device = "cuda:{}".format(self.gpu_index)

        scores = {
            'model_type':       [],
            'num_params':       [],
            'ave_accuracy':     [],
            'ave_f1':           [],
            'ave_precision':    [],
            'ave_recall':       [],
            'ave_jaccard':      [],
            'elapsed_time':     []
        }

        for model_i, model_cfg in enumerate(self.models):
            model_type = model_cfg['type']
            model_file = model_cfg['file']

            print(f"Benchmarking model {model_type} ({model_file})")

            state = torch.load(
                model_file,
                map_location=self.device
            )

            model = load_model_for_inference(
                self.in_channels,
                self.out_channels,
                model_type,
                self.device,
                state['state_dict']
            )

            test_images = sorted(glob.glob("{}/*".format(self.input_img_dir)))
            test_masks  = sorted(glob.glob("{}/*".format(self.input_mask_dir)))

            dim = (self.img_width, self.img_height)

            num_images = len(test_images)

            ave_accuracy    = 0.0
            ave_f1          = 0.0
            ave_precision   = 0.0
            ave_recall      = 0.0
            ave_jaccard     = 0.0

            start_time = time.time()

            for i in range(num_images):
                image_file  = test_images[i]
                mask_file   = test_masks[i]

                img  = get_image(image_file, dim)
                mask = get_mask(mask_file, dim)

                prediction = get_predicted_img(img, model, device=self.device)

                mask_vectorized = mask.ravel().astype(int)
                prediction_vectorized = prediction.ravel().astype(int)

                accuracy    = accuracy_score(mask_vectorized, prediction_vectorized)
                f1          = f1_score(mask_vectorized, prediction_vectorized, average='macro', zero_division=1)
                precision   = precision_score(mask_vectorized, prediction_vectorized, average='macro', zero_division=1)
                recall      = recall_score(mask_vectorized, prediction_vectorized, average='macro', zero_division=1) # sensitivity
                jaccard     = jaccard_score(mask_vectorized, prediction_vectorized, labels=range(self.out_channels), average='macro')

                ave_accuracy += accuracy
                ave_f1 += f1
                ave_precision += precision
                ave_recall += recall
                ave_jaccard += jaccard

            end_time = time.time()

            elapsed_time = end_time - start_time
            elapsed_time = round(elapsed_time, 4)

            ave_accuracy    = ave_accuracy / num_images
            ave_f1          = ave_f1 / num_images
            ave_precision   = ave_precision / num_images
            ave_recall      = ave_recall / num_images
            ave_jaccard     = ave_jaccard / num_images

            scores['model_type'].append(model_type)
            scores['num_params'].append(count_parameters(model))
            scores['ave_accuracy'].append(ave_accuracy)
            scores['ave_f1'].append(ave_f1)
            scores['ave_precision'].append(ave_precision)
            scores['ave_recall'].append(ave_recall)
            scores['ave_jaccard'].append(ave_jaccard)
            scores['elapsed_time'].append(elapsed_time)

        df_results = pd.DataFrame(scores)
        print(df_results)
