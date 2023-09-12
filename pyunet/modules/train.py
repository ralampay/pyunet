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
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.loss_functions import dice_loss, tversky_loss, FocalLoss, sym_unified_focal_loss
from lib.utils import get_image, get_mask, get_predicted_img, dice_score, initialize_model

class Train:
    def __init__(self, params={}, seed=0):
        if seed >= 0:
            torch.manual_seed(seed)

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

        self.writer = SummaryWriter()

        self.model = None

    def execute(self):
        print("Training model...")

        print("input_img_dir: {}".format(self.input_img_dir))
        print("input_mask_dir: {}".format(self.input_mask_dir))

        if self.device == 'cuda':
            print("CUDA Device: {}".format(torch.cuda.get_device_name(self.gpu_index)))
            self.device = "cuda:{}".format(self.gpu_index)

        self.model = initialize_model(
            self.in_channels,
            self.out_channels,
            self.model_type,
            self.device
        )

        print(self.model)

        if self.cont and os.path.exists(self.model_file):
            state = torch.load(
                self.model_file, 
                map_location=self.device
            )

            self.model.load_state_dict(state['state_dict'])
            self.model.optimizer     = state['optimizer']

        if self.loss_type == 'CE':
            loss_fn = nn.CrossEntropyLoss()
        elif self.loss_type == 'DL':
            loss_fn = dice_loss
        elif self.loss_type == 'TL':
            loss_fn = tversky_loss
        elif self.loss_type == 'FL':
            loss_fn = FocalLoss()
        elif self.loss_type == 'DP':
            loss_fn = depth_loss
        elif self.loss_type == 'MSE':
            loss_fn = nn.MSELoss()
        else:
            raise ValueError("Unsupported loss_type {}".format(self.loss_type))

        print("Loss Type: {}".format(self.loss_type))

        optimizer   = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scaler      = torch.cuda.amp.GradScaler()

        train_ds = CustomDataset(
            image_dir=self.input_img_dir,
            mask_dir=self.input_mask_dir,
            img_width=self.img_width,
            img_height=self.img_height,
            num_classes=self.out_channels
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )

        for epoch in range(self.epochs):
            print("Epoch: {}".format(epoch+1))

            ave_loss, ave_accuracy, ave_f1, ave_precision, ave_recall, ave_specificity = self.train_fn(
                train_loader, 
                self.model, 
                optimizer, 
                loss_fn, 
                scaler,
                test_img_dir=self.test_img_dir,
                test_mask_dir=self.test_mask_dir
            )

            # write loss to tensorboard
            self.writer.add_scalar(f"Loss ({self.model_type}-{self.loss_type})", ave_loss, epoch+1)

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
                'state_dict': self.model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            torch.save(state, self.model_file)

            self.writer.flush()


    def train_fn(self, loader, model, optimizer, loss_fn, scaler, test_img_dir=None, test_mask_dir=None):
        loop = tqdm(loader)

        ave_loss = 0.0
        count = 0

        for batch_idx, (data, targets) in enumerate(loop):
            data    = data.float().to(device=self.device)
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

            # Write to tensorboard

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
    def __init__(self, image_dir, mask_dir, img_width, img_height, num_classes):
        self.image_dir      = image_dir
        self.mask_dir       = mask_dir 
        self.img_width      = img_width
        self.img_height     = img_height
        self.images         = sorted(os.listdir(image_dir))
        self.images_masked  = sorted(os.listdir(mask_dir))
        self.num_classes    = num_classes

        self.dim = (img_width, img_height)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path    = os.path.join(self.image_dir, self.images[index])
        mask_path   = os.path.join(self.mask_dir, self.images_masked[index])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        original_img    = (cv2.resize(img, self.dim) / 255).transpose((2, 0, 1))
        masked_img      = (cv2.resize(cv2.imread(mask_path, 0), self.dim))

        x = torch.Tensor(original_img)
        y = torch.Tensor(masked_img)

        return x, y
