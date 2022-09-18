import sys
import os
import torch
import cv2
import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class SamplePair:
    def __init__(self, params={}):
        self.params = params

        self.img_width  = params.get('img_width')
        self.img_height = params.get('img_height')
        
        self.input_img_dir  = params.get('input_img_dir')
        self.input_mask_dir = params.get('input_mask_dir')

    def execute(self):
        print("Sampling pair...")

        print("input_img_dir: {}".format(self.input_img_dir))
        print("input_mask_dir: {}".format(self.input_mask_dir))

        img_list = sorted(os.listdir(self.input_img_dir))
        msk_list = sorted(os.listdir(self.input_mask_dir))

        num_images = len(os.listdir(self.input_img_dir))

        sampled_index = random.randint(0, num_images - 1)

        img_path    = os.path.join(self.input_img_dir, img_list[sampled_index])
        mask_path   = os.path.join(self.input_mask_dir, msk_list[sampled_index])

        print(img_path)
        print(mask_path)

        img_for_plot = cv2.imread(img_path, 1)
        img_for_plot = cv2.cvtColor(img_for_plot, cv2.COLOR_BGR2RGB)

        mask_for_plot = cv2.imread(mask_path, 0)

        dim = (self.img_width, self.img_height)

        img_for_plot    = cv2.resize(img_for_plot, dim)
        mask_for_plot   = cv2.resize(mask_for_plot, dim)

        plt.figure(figsize=(12, 8))
        plt.subplot(121)
        plt.imshow(img_for_plot)
        plt.title('Image')
        plt.subplot(122)
        plt.imshow(mask_for_plot, cmap='gray')
        plt.title('Mask')
        plt.show()
