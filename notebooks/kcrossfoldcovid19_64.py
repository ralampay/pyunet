#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import cv2
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import jaccard_score
import copy

sys.path.append(os.path.join('', '../pyunet/lib'))
from unet import UNet
from loss_functions import dice_loss, tversky_loss

device          = 'cuda'
in_channels     = 3
out_channels    = 4
image_dir       = "./images/covid19ctscan/images"
mask_dir        = "./images/covid19ctscan/masks"
learning_rate   = 0.0001
img_width       = 64
img_height      = 64
gpu_index       = 0
batch_size      = 2
dim             = (img_width, img_height)

# Experimental configurations
experiments = [
    { 'loss_type': 'CE', 'is_normalized': True, 'model_file': './models/covid19ctscan-{}-{}-CE-true.pth'.format(img_width, img_height) },
    { 'loss_type': 'CE', 'is_normalized': False, 'model_file': './models/covid19ctscan-{}-{}-CE-false.pth'.format(img_width, img_height) },
    { 'loss_type': 'DL', 'is_normalized': True, 'model_file': './models/covid19ctscan-{}-{}-DL-true.pth'.format(img_width, img_height) },
    { 'loss_type': 'DL', 'is_normalized': False, 'model_file': './models/covid19ctscan-{}-{}-DL-false.pth'.format(img_width, img_height) },
    { 'loss_type': 'TL', 'is_normalized': True, 'model_file': './models/covid19ctscan-{}-{}-TL-true.pth'.format(img_width, img_height) },
    { 'loss_type': 'TL', 'is_normalized': False, 'model_file': './models/covid19ctscan-{}-{}-TL-false.pth'.format(img_width, img_height) },
]

# Number of partitions
k = 10

# Number of epochs per training session
epochs = 100


# In[2]:


class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, indices, dim):
        self.image_paths    = image_paths
        self.mask_paths     = mask_paths
        self.indices        = indices
        self.dim            = dim

        self.image_dataset  = []
        self.mask_dataset   = []

        for i in range(len(indices)):
            self.image_dataset.append(self.image_paths[indices[i]])
            self.mask_dataset.append(self.mask_paths[indices[i]])

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, index):
        x = torch.Tensor(
            (
                cv2.resize(
                    cv2.imread(self.image_dataset[index]),
                    self.dim
                ) / 255
            ).transpose((2, 0, 1))
        )

        y = torch.Tensor(
            cv2.resize(
                cv2.imread(self.mask_dataset[index], 0),
                self.dim
            )
        )

        return x, y


# In[3]:


def train_fn(loader, model, optimizer, loss_fn, scaler, device):
    loop = tqdm(loader)

    ave_loss = 0.0
    count = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data    = data.to(device)
        targets = targets.long().to(device=device)

        # Forward
        predictions = model.forward(data)

        loss = loss_fn(predictions, targets)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update tqdm
        loop.set_postfix(loss=loss.item())

        ave_loss += loss.item()
        count += 1

    ave_loss = ave_loss / count
    
    return ave_loss


# In[4]:


from unicodedata import is_normalized



image_paths = list(map(lambda o: os.path.join(image_dir, o), sorted(os.listdir(image_dir))))
mask_paths  = list(map(lambda o: os.path.join(mask_dir, o), sorted(os.listdir(mask_dir))))

num_items       = len(image_paths)
len_partition   = int(num_items / k)

print("Len Partition: {}".format(len_partition))

indices = list(range(num_items))

index = 0

aggregated_scores = []

for i in range(k):
    device = 'cuda'
    scores = []

    validation_indices  = np.array(indices[index:index + len_partition])
    training_indices    = np.delete(indices, validation_indices)

    index = index + len_partition
    if device == 'cuda':
        print("CUDA Device: {}".format(torch.cuda.get_device_name(gpu_index)))
        device = "cuda:{}".format(gpu_index)

    print("Device: {}".format(device))

    # Loop through each experiment
    for experiment in experiments:
        is_normalized   = experiment.get('is_normalized')
        loss_type       = experiment.get('loss_type')
        model_file      = experiment.get('model_file')

        model = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            is_normalized=is_normalized
        ).to(device)

        if loss_type == 'CE':
            loss_fn = nn.CrossEntropyLoss()
        elif loss_type == 'DL':
            loss_fn = dice_loss
        elif loss_type == 'TL':
            loss_fn = tversky_loss

        print("K: {} Loss Fn: {} Is Normalized: {} Model File: {}".format(i+1, loss_type, is_normalized, model_file))
        optimizer   = optim.Adam(model.parameters(), lr=learning_rate)
        scaler      = torch.cuda.amp.GradScaler()

        train_ds = CustomDataset(
            image_paths=image_paths,
            mask_paths=mask_paths,
            indices=training_indices,
            dim=dim
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )

        losses = []

        for epoch in range(epochs):
            print("Epoch: {}".format(epoch))
            ave_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, device)

            print("Ave Loss: {}".format(ave_loss))
            print("Saving file to {}".format(model_file))

            state = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'out_channels': out_channels
            }

            losses.append(ave_loss)

            torch.save(state, model_file)

        x_validation = []
        y_validation = []

        for j in range(len(validation_indices)):
            x_validation.append(
                cv2.cvtColor(
                    cv2.resize(
                        cv2.imread(
                            image_paths[validation_indices[j]],
                            1
                        ),
                        dim
                    ),
                    cv2.COLOR_BGR2RGB    
                ).transpose((2, 0, 1))
            )

            y_validation.append(
                cv2.resize(
                    cv2.imread(
                        mask_paths[validation_indices[j]],
                        0
                    ),
                    dim
                )
            )

        x_validation = torch.FloatTensor(np.array(x_validation)).to(device)
        y_validation = torch.FloatTensor(np.array(y_validation)).to(device)

        predictions = torch.argmax(model.forward(x_validation), 1)

        macro_scores = []
        label_scores = []

        for prediction_index in range(len(x_validation)):
            prediction = predictions[prediction_index]

            target = y_validation[0]

            target = target.detach().cpu().numpy().ravel()
            prediction = prediction.detach().cpu().numpy().ravel()

            macro_score = jaccard_score(target, prediction, average='macro')
            label_score = jaccard_score(target, prediction, average=None)

            print(label_score)

            macro_scores.append(macro_score)
            label_scores.append(copy.deepcopy(label_score))

        scores.append({
            'k': i+1,
            'experiment': experiment,
            'macro_scores': macro_scores,
            'label_scores': copy.deepcopy(label_scores),
            'losses': copy.deepcopy(losses),
            'ave_macro_score': sum(macro_scores) / len(macro_scores)
        })

    aggregated_scores.append(copy.deepcopy(scores))

#    print("Validation Indices")
#    print(validation_indices)
#
#    print("Training Indices")
#    print(training_indices)
#
#    print("================================")
#    validation_image_paths = image_paths[index:index + len_partition]
#    validation_mask_paths = mask_paths[index:index + len_partition]
#
#    print("Validation Image Paths:")
#    print(validation_image_paths)
#
#    print("Validation Mask Paths:")
#    print(validation_mask_paths)


# In[ ]:


# Save json
import codecs, json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

aggregated_scores = np.array(aggregated_scores).tolist()
file_path = './aggregate_scores_64.json'

json.dump(aggregated_scores, codecs.open(file_path, 'w', encoding='utf-8'),
    separators=(',', ':'),
    sort_keys=True,
    indent=2,
    cls=NpEncoder)

aggregated_scores

