import torch
import json
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

sys.path.append(os.path.join('', '../pyunet/lib'))
from unet import UNet

device          = 'cuda'
in_channels     = 3
out_channels    = 5
image_dir       = "/home/ralampay/workspace/pysplitter/sample-images"
mask_dir        = "/home/ralampay/workspace/pysplitter/sample-masks"
model_file      = "./models/landsat-ai-256.pth"
learning_rate   = 0.0001
img_width       = 256
img_height      = 256
epochs          = 150
gpu_index       = 0
features        = [64, 128, 256, 512]
batch_size      = 1
dim             = (img_width, img_height)

# Number of partitions
k = 9


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


scores = []

image_paths = list(map(lambda o: os.path.join(image_dir, o), sorted(os.listdir(image_dir))))
mask_paths  = list(map(lambda o: os.path.join(mask_dir, o), sorted(os.listdir(mask_dir))))

num_items       = len(image_paths)
len_partition   = int(num_items / k)

print("Len Partition: {}".format(len_partition))

indices = list(range(num_items))

index = 0

for i in range(k):
    device = 'cuda'

    validation_indices  = np.array(indices[index:index + len_partition])
    training_indices    = np.delete(indices, validation_indices)

    index = index + len_partition
    if device == 'cuda':
        print("CUDA Device: {}".format(torch.cuda.get_device_name(gpu_index)))
        device = "cuda:{}".format(gpu_index)

    print("Device: {}".format(device))

    model = UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        features=features
    ).to(device)

    loss_fn     = nn.CrossEntropyLoss()
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

    for epoch in range(epochs):
        print("Epoch: {}".format(epoch))
        ave_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, device)

        print("Ave Loss: {}".format(ave_loss))

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

    device = 'cpu'
    model = model.to(device)
    x_validation = torch.FloatTensor(np.array(x_validation)).to(device)
    y_validation = torch.FloatTensor(np.array(y_validation)).to(device)

    predictions = torch.argmax(model.forward(x_validation), 1).detach().numpy().astype(np.int32)

    macro_scores = []
    label_scores = []

    for prediction_index in range(len(x_validation)):
        prediction = predictions[prediction_index]

        target = y_validation[0].detach().numpy().astype(np.int32)

        macro_score = jaccard_score(target.ravel(), prediction.ravel(), average='macro')
        label_score = jaccard_score(target.ravel(), prediction.ravel(), average=None)

        macro_scores.append(macro_score)
        label_scores.append(label_scores)

    scores.append({
        'k': i,
        'macro_scores': macro_scores,
        'ave_macro_score': sum(macro_scores) / len(macro_scores)
    })

print(scores)

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

