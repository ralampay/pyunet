import sys
import os 
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.unet import UNet

class Train:
    def __init__(self, params={}):
        self.params = params

        self.img_width      = params.get('img_height')
        self.img_height     = params.get('img_height')
        self.device         = params.get('device')
        self.gpu_index      = params.get('gpu_index')
        self.input_img_dir  = params.get('input_img_dir')
        self.input_mask_dir = params.get('input_mask_dir')
        self.epochs         = params.get('epochs')
        self.learning_rate  = params.get('learning_rate')
        self.model_file     = params.get('model_file')
        self.batch_size     = params.get('batch_size')
        self.in_channels    = params.get('in_channels') or 3
        self.out_channels   = params.get('out_channels') or 3
        self.features       = params.get('features') or [64, 128, 256, 512]
        self.cont           = params.get('cont') or False

    def execute(self):
        print("Training model...")

        print("input_img_dir: {}".format(self.input_img_dir))
        print("input_mask_dir: {}".format(self.input_mask_dir))

        if self.device == 'cuda':
            print("CUDA Device: {}".format(torch.cuda.get_device_name(self.gpu_index)))
            self.device = "cuda:{}".format(self.gpu_index)

        model = UNet(
            in_channels=self.in_channels, 
            out_channels=self.out_channels,
            features=self.features
        ).to(self.device)

        if self.cont:
            state = torch.load(self.model_file)
            model.load_state_dict(state['state_dict'])
            model.optimizer     = state['optimizer']
            model.in_channels   = self.in_channels
            model.out_channels  = state['out_channels']
            model.features      = self.features

        loss_fn     = nn.CrossEntropyLoss()
        optimizer   = optim.Adam(model.parameters(), lr=self.learning_rate)
        scaler      = torch.cuda.amp.GradScaler()

        train_ds = CustomDataset(
            image_dir=self.input_img_dir,
            mask_dir=self.input_mask_dir,
            img_width=self.img_width,
            img_height=self.img_height,
            n_classes=self.out_channels,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )

        for epoch in range(self.epochs):
            print("Epoch: {}".format(epoch))
            ave_loss = self.train_fn(train_loader, model, optimizer, loss_fn, scaler)

            print("Ave Loss: {}".format(ave_loss))

            # Save model after every epoch
            print("Saving model to {}...".format(self.model_file))

            state = {
                'params': self.params,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'out_channels': self.out_channels
            }

            torch.save(state, self.model_file)


    def train_fn(self, loader, model, optimizer, loss_fn, scaler):
        loop = tqdm(loader)

        ave_loss = 0.0
        count = 0

        for batch_idx, (data, targets) in enumerate(loop):
            data    = data.to(device=self.device)
            targets = targets.long().to(device=self.device)

            # Forward
            predictions = model.forward(data)

            loss = loss_fn(predictions, targets)

            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update tqdm
            loop.set_postfix(loss=loss.item())

            ave_loss += loss.item()
            count += 1

        ave_loss = ave_loss / count

        return ave_loss
        

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_width, img_height, n_classes):
        self.image_dir      = image_dir
        self.mask_dir       = mask_dir 
        self.img_width      = img_width
        self.img_height     = img_height
        self.images         = sorted(os.listdir(image_dir))
        self.images_masked  = sorted(os.listdir(mask_dir))
        self.n_classes      = n_classes

        self.dim = (img_width, img_height)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path    = os.path.join(self.image_dir, self.images[index])
        mask_path   = os.path.join(self.mask_dir, self.images_masked[index])

        original_img    = (cv2.resize(cv2.imread(img_path), self.dim) / 255).transpose((2, 0, 1))
        masked_img      = (cv2.resize(cv2.imread(mask_path, 0), self.dim))

        return torch.Tensor(original_img), torch.Tensor(masked_img)
