import sys
import os
import cv2
import numpy as np

def grayscale_to_hex(grayscale_value):
    # Convert the grayscale value to a hex string
    hex_string = hex(grayscale_value)[2:]

    # Add leading zeros if necessary
    while len(hex_string) < 2:
        hex_string = "0" + hex_string

    # Return the full hex string
    return "#" + hex_string + hex_string + hex_string

class ExtractUniqueGray:
    def __init__(self, params={}):
        self.grayscale_values = []

        self.params         = params
        self.input_img_dir  = params.get('input_img_dir') 
        self.img_suffix     = params.get('img_suffix') or 'png'

    def execute(self):
        for filename in os.listdir(self.input_img_dir):
            print("Reading file {}...".format(filename))
            img = cv2.imread(os.path.join(self.input_img_dir, filename), 0)

            unique_values = np.unique(img)
            print("Unique values:")
            print(unique_values)

            set1 = set(self.grayscale_values)
            set2 = set(unique_values)

            difference = list(set2.difference(set1))

            for val in difference:
                self.grayscale_values.append(val)

        self.grayscale_values = sorted(self.grayscale_values)

        print("Unique Grayscale Values:")
        print(self.grayscale_values)
