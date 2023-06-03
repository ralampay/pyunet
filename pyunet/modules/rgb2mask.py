import sys
import os
import torch
import cv2
import time
import numpy as np
import json

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
        # Display labels
        print(f"Dataset: {self.config['title']}")
        for label in self.labels:
            print(f"Name: {label['name']} Color: ({label['color'][0]}, {label['color'][1]} {label['color'][2]})")

        image = cv2.imread(self.image_file)
        image = cv2.resize(image, self.dim)

        rows, cols, _ = image.shape

        original_dim = (cols, rows)

        result = image

        cv2.imshow("result", cv2.resize(result, original_dim))
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 
