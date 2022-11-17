# In[ ]:


import json
import pandas as pd
import copy
import glob
import cv2
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
sys.path.append(os.path.join('./','../pyunet'))
from lib.unet import UNet
from modules.train import Train
import torch
from lib.utils import get_image, get_mask, get_predicted_img, dice_score
import glob
from sklearn.model_selection import train_test_split
import shutil
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


# In[ ]:


img_dir  = "/home/ralampay/workspace/pycocosegmentor/images/coco2017car/images/"
mask_dir = "/home/ralampay/workspace/pycocosegmentor/images/coco2017car/masks/"

# In[ ]:


img_height     = 256
img_width      = 256
device         = 'cuda'
gpu_index      = 0
input_img_dir  = img_dir
input_mask_dir = mask_dir
model_file     = "coco-car-unet-rd.pth"
epochs         = 100
learning_rate  = 0.0001
in_channels    = 3
out_channels   = 2
batch_size     = 10
loss_type      = 'CE'
cont           = True

params = {
    'img_height':     img_height,
    'img_width':      img_width,
    'device':         device,
    'gpu_index':      gpu_index,
    'input_img_dir':  input_img_dir,
    'input_mask_dir': input_mask_dir,
    'epochs':         epochs,
    'learning_rate':  learning_rate,
    'in_channels':    in_channels,
    'out_channels':   out_channels,
    'loss_type':      loss_type,
    'batch_size':     batch_size,
    'model_file':     model_file,
    'test_img_dir':   None,
    'test_mask_dir':  None,
    'cont':           cont,
    'model_type':     'unet_rd'
}

cmd = Train(params=params)

cmd.execute()
