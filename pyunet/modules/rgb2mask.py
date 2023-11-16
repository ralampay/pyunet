import sys
import os
import torch
import cv2
import time
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.utils import rgb2mask

class Rgb2Mask:
    def __init__(self, params={}):
        self.params = params

        self.img_width      = params.get('img_width')
        self.img_height     = params.get('img_height')
        self.config_file    = params.get('config_file')
        self.image_file     = params.get('image_file')
        self.dim            = (self.img_width, self.img_height)

        self.config = json.load(open(self.config_file))
        self.labels = self.config["labels"]

    def execute(self):
        print(f"Dimension: {self.dim}")
        # Display labels
        colors = []
        print(f"Dataset: {self.config['title']}")
        for label in self.labels:
            print(f"Name: {label['name']} Color: ({label['color'][0]}, {label['color'][1]} {label['color'][2]})")
            colors.append(label['color'])

        print("Colors:")
        print(colors)

        self.image  = cv2.imread(self.image_file)
        self.result = rgb2mask(colors, self.image)

        plt.figure(figsize=(8, 4))
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.title('Image')
        plt.subplot(122)
        plt.imshow(self.result, cmap='gray')
        plt.title('Mask')

        plt.show()
