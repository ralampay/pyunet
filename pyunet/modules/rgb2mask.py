import sys
import os
import torch
import cv2
import time
import numpy as np
import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

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

        image = cv2.imread(self.image_file)

        rows, cols, _ = image.shape

        original_dim = (cols, rows)

        self.result = np.zeros((rows, cols))

        for y in range(0, rows):
            for x in range(0, cols):
                b, g, r = (image[y, x])

                for i in range(len(colors)):
                    if colors[i] == [r, g, b]:
                        self.result[y, x] = i

        plt.figure(figsize=(12, 8))
        plt.subplot(121)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Image')
        plt.subplot(122)
        plt.imshow(self.result, cmap='gray')
        plt.title('Mask')

        plt.show()
