import torch
from torch import tensor
import cv2
import glob
import numpy as np

def load_image_tensors(input_img_dir, img_width, img_height):
    images = []

    ext = ['png', 'jpg', 'gif', 'tiff', 'tif']

    files = []
    [files.extend(glob.glob(input_img_dir + '/*.' + e)) for e in ext]

    dim = (img_width, img_height)

    images = np.array([cv2.resize(cv2.imread(f), dim) for f in files])

    images = images / 255

    x = []

    for img in images:
        result = img.transpose((2, 0, 1))
        x.append(result)

    return torch.tensor(x).float()
