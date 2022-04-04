import sys
import os
import torch
import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class GenerateTiff:
    def __init__(self, params={}):
        self.params = params

        self.input_img_dir   = params.get('input_img_dir')
        self.output_img_dir  = params.get('output_img_dir')
        self.unique_values   = params.get('unique_values')

    def execute(self):

        for filename in os.listdir(self.input_img_dir):
            img = cv2.imread(os.path.join(self.input_img_dir, filename))

            # Convert OpenCV BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if img is not None:
                tiff_filename = filename.replace(".png", ".tiff")
                print("Processing {}...".format(filename))

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                tiff = self.convert_to_labeled_tiff(gray)

                cv2.imwrite(os.path.join(self.output_img_dir, tiff_filename), tiff)

                print("Saving to {}...".format(tiff_filename))

    def convert_to_labeled_tiff(self, img_grayscale):
        rows, cols = img_grayscale.shape[:2]
        tiff = np.zeros((rows, cols), dtype=np.uint8)

        for r in range(rows):
            for c in range(cols):
                for idx, val in enumerate(self.unique_values):
                    if val == img_grayscale[r, c]:
                        tiff[r, c] = idx

        return tiff
