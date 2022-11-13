import sys
import os 
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
import glob
from sklearn.metrics import jaccard_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.unet import UNet
from lib.unet_rd import UNetRd
from lib.loss_functions import dice_loss, tversky_loss
from lib.utils import get_image, get_mask, get_predicted_img, dice_score

class Train:
    def __init__(self, params={}):
        self.params = params

        self.img_width              = params.get('img_height')
        self.img_height             = params.get('img_height')
        self.device                 = params.get('device')
        self.gpu_index              = params.get('gpu_index')
        self.input_img_dir          = params.get('input_img_dir')
        self.input_mask_dir         = params.get('input_mask_dir')
        self.epochs                 = params.get('epochs')
        self.learning_rate          = params.get('learning_rate')
        self.model_file             = params.get('model_file')
        self.batch_size             = params.get('batch_size')
        self.in_channels            = params.get('in_channels') or 3
        self.out_channels           = params.get('out_channels') or 3
        self.cont                   = params.get('cont') or False
        self.loss_type              = params.get('loss_type') or 'CE'
        self.model_type             = params.get('model_type') or 'unet'

        self.test_img_dir   = params.get('test_img_dir') or None
        self.test_mask_dir  = params.get('test_mask_dir') or None

        self.accuracies     = []
        self.f1s            = []
        self.precisions     = []
        self.recalls        = []
        self.specificities  = []
        self.losses         = []

        self.model = None

    def execute(self):
        print("Training model...")

        print("input_img_dir: {}".format(self.input_img_dir))
        print("input_mask_dir: {}".format(self.input_mask_dir))

        if self.device == 'cuda':
            print("CUDA Device: {}".format(torch.cuda.get_device_name(self.gpu_index)))
            self.device = "cuda:{}".format(self.gpu_index)

        if self.model_type == 'unet':
            self.model = UNet(
                in_channels=self.in_channels, 
                out_channels=self.out_channels
            ).to(self.device)
        elif self.model_type == 'unet_rd':
            self.model = UNetRd(
                in_channels=self.in_channels, 
                out_channels=self.out_channels
            ).to(self.device)

        print(self.model)

        if self.cont:
            state = torch.load(
                self.model_file, 
                map_location=self.device
            )

            self.model.load_state_dict(state['state_dict'])
            self.model.optimizer     = state['optimizer']
            self.model.in_channels   = self.in_channels
            self.model.out_channels  = self.out_channels

        if self.loss_type == 'CE':
            loss_fn = nn.CrossEntropyLoss()
        elif self.loss_type == 'DL':
            loss_fn = dice_loss
        elif self.loss_type == 'TL':
            loss_fn = tversky_loss
        else:
            loss_fn = nn.CrossEntropyLoss()

        print("Loss Type: {}".format(self.loss_type))


        optimizer   = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scaler      = torch.cuda.amp.GradScaler()

        train_ds = CustomDataset(
            image_dir=self.input_img_dir,
            mask_dir=self.input_mask_dir,
            img_width=self.img_width,
            img_height=self.img_height
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )

        for epoch in range(self.epochs):
            print("Epoch: {}".format(epoch))

            ave_loss, ave_accuracy, ave_f1, ave_precision, ave_recall, ave_specificity = self.train_fn(
                train_loader, 
                self.model, 
                optimizer, 
                loss_fn, 
                scaler,
                test_img_dir=self.test_img_dir,
                test_mask_dir=self.test_mask_dir
            )

            self.losses.append(ave_loss)

            print("Ave Loss: {}".format(ave_loss))

            if self.test_img_dir is not None and self.test_mask_dir is not None:
                self.accuracies.append(ave_accuracy)
                print("Ave Accuracy: {}".format(ave_accuracy))

                self.f1s.append(ave_f1)
                print("Ave F1: {}".format(ave_f1))

                self.precisions.append(ave_precision)
                print("Ave Precision: {}".format(ave_precision))

                self.recalls.append(ave_recall)
                print("Ave Recall: {}".format(ave_recall))

                self.specificities.append(ave_specificity)
                print("Ave Specificity: {}".format(ave_specificity))

            # Save model after every epoch
            print("Saving model to {}...".format(self.model_file))

            state = {
                'params': self.params,
                'state_dict': self.model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'out_channels': self.out_channels,
                'is_normalized': self.is_normalized,
                'is_residual': self.is_residual,
                'double_skip': self.double_skip
            }

            torch.save(state, self.model_file)


    def train_fn(self, loader, model, optimizer, loss_fn, scaler, test_img_dir=None, test_mask_dir=None):
        loop = tqdm(loader)

        ave_loss = 0.0
        count = 0

        for batch_idx, (data, targets) in enumerate(loop):
            data    = data.to(device=self.device)
            targets = targets.long().to(device=self.device)

            # Forward
            predictions = model.forward(data)

            loss = loss_fn(predictions, targets)
            #loss = self.dice_loss(predictions, targets)

            # Backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update tqdm
            loop.set_postfix(loss=loss.item())

            ave_loss += loss.item()
            count += 1

        # Compute the accuracies if test_img_dir and test_mask_dir are present
        ave_accuracy    = None
        ave_f1          = None
        ave_precision   = None
        ave_recall      = None
        ave_specificity = None

        if test_img_dir is not None and test_mask_dir is not None:
            test_images = sorted(glob.glob("{}/*".format(test_img_dir)))
            test_masks  = sorted(glob.glob("{}/*".format(test_mask_dir)))

            dim = (self.img_width, self.img_height)

            num_images = len(test_images)

            ave_accuracy    = 0.0
            ave_f1          = 0.0
            ave_precision   = 0.0
            ave_recall      = 0.0
            ave_specificity = 0.0

            for i in range(num_images):
                image_file  = test_images[i]
                mask_file   = test_masks[i]

                img  = get_image(image_file, dim)
                mask = get_mask(mask_file, dim)

                prediction = get_predicted_img(img, model, device=self.device)

                mask_vectorized = mask.ravel().astype(int)
                prediction_vectorized = prediction.ravel().astype(int)

                accuracy    = accuracy_score(mask_vectorized, prediction_vectorized)
                f1          = f1_score(mask_vectorized, prediction_vectorized, average='macro', zero_division=1)
                precision   = precision_score(mask_vectorized, prediction_vectorized, average='macro', zero_division=1)
                recall      = recall_score(mask_vectorized, prediction_vectorized, average='macro', zero_division=1) # sensitivity
                specificity = recall_score(mask_vectorized, prediction_vectorized, labels=range(self.out_channels), average='macro', zero_division=1)

                ave_accuracy += accuracy
                ave_f1 += f1
                ave_precision += precision
                ave_recall += recall
                ave_specificity += specificity

            ave_accuracy    = ave_accuracy / num_images
            ave_f1          = ave_f1 / num_images
            ave_precision   = ave_precision / num_images
            ave_recall      = ave_recall / num_images
            ave_specificity = ave_specificity / num_images

        ave_loss = ave_loss / count

        return ave_loss, ave_accuracy, ave_f1, ave_precision, ave_recall, ave_specificity

        

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, img_width, img_height):
        self.image_dir      = image_dir
        self.mask_dir       = mask_dir 
        self.img_width      = img_width
        self.img_height     = img_height
        self.images         = sorted(os.listdir(image_dir))
        self.images_masked  = sorted(os.listdir(mask_dir))

        self.dim = (img_width, img_height)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path    = os.path.join(self.image_dir, self.images[index])
        mask_path   = os.path.join(self.mask_dir, self.images_masked[index])

        original_img    = (cv2.resize(cv2.imread(img_path), self.dim) / 255).transpose((2, 0, 1))
        masked_img      = (cv2.resize(cv2.imread(mask_path, 0), self.dim))

        return torch.Tensor(original_img), torch.Tensor(masked_img)
