import sys
import os
import torch
import cv2
import numpy as np

class GenerateTiff:
    def __init__(self, params={}):
        self.params = params

        self.input_img_dir  = params.get('input_img_dir')
        self.output_img_dir = params.get('output_img_dir')
        self.unique_values  = params.get('unique_values')
        self.img_suffix     = params.get('img_suffix') or 'jpg'

    def execute(self):
        print('Mode: GenerateTiff')
        print('input_img_dir: {}'.format(self.input_img_dir))
        print('output_img_dir: {}'.format(self.output_img_dir))

        for filename in os.listdir(self.input_img_dir):
            img = cv2.imread(os.path.join(self.input_img_dir, filename))

            # Convert OpenCV BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if img is not None:
                tiff_filename = filename.replace(".{}".format(self.img_suffix), ".tiff")
                print("Processing {}/{}...".format(self.input_img_dir, filename))

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

                tiff = self.convert_to_labeled_tiff(gray)

                print("Saving to {}/{}...".format(self.output_img_dir, tiff_filename))
                cv2.imwrite(os.path.join(self.output_img_dir, tiff_filename), tiff)

    def convert_to_labeled_tiff(self, img_grayscale):
        rows, cols = img_grayscale.shape[:2]
        tiff = np.zeros((rows, cols), dtype=np.uint8)

        for r in range(rows):
            for c in range(cols):
                for idx, val in enumerate(self.unique_values):
                    px_val = img_grayscale[r, c]

                    # If binary, then treat all values > 0 as white
                    if len(self.unique_values) <= 2 and px_val > 0:
                        px_val = 255

                    if val == px_val:
                        tiff[r, c] = idx

        return tiff
