import cv2
import torch
import numpy as np

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.utils import load_model_for_inference

class SampleFrame:
    def __init__(self, params):
        self.device         = params.get('device') or 'cuda'
        self.img_width      = params.get('img_width')
        self.img_height     = params.get('img_height')
        self.input_img      = params.get('input_img')
        self.model_type     = params.get('model_type')
        self.model_file     = params.get('model_file')
        self.in_channels    = params.get('in_channels') or 3
        self.out_channels   = params.get('out_channels') or 2

        # Input dimension size
        self.dim = (self.img_width, self.img_height)

        # TODO: Parameterize this?
        self.frame_width    = 1200
        self.frame_height   = 600

        self.img_layout = np.zeros((self.frame_height, self.frame_width, 3), np.uint8)

        # BGR
        # Drawing parameters
        self.border_color       = (0, 0, 255)
        self.thickness          = 3 
        self.subframe_padding   = 32

        self.mid = self.frame_width // 2
        self.display_dim = (self.mid - self.subframe_padding * 2, self.frame_height - self.subframe_padding * 2)

    def draw_layout(self):
        # Draw line in the middle
        start_point = (self.mid, 0)
        end_point   = (self.mid, self.frame_height)
        cv2.line(self.img_layout, start_point, end_point, self.border_color, self.thickness)

        # Draw left subframe
        start_point = (self.subframe_padding, self.subframe_padding)
        end_point   = (self.mid - self.subframe_padding, self.frame_height - self.subframe_padding)
        cv2.rectangle(self.img_layout, start_point, end_point, self.border_color, self.thickness)

        # Draw right subframe
        start_point = (self.mid + self.subframe_padding, self.subframe_padding)
        end_point   = (self.frame_width - self.subframe_padding, self.frame_height - self.subframe_padding)
        cv2.rectangle(self.img_layout, start_point, end_point, self.border_color, self.thickness)

    def overlay(self, orig, segmented):
        # Overlay left
        y_offset = self.subframe_padding
        x_offset = self.subframe_padding

        print(segmented.shape)
        # Overlay left
        self.img_layout[self.subframe_padding:self.subframe_padding+orig.shape[0], self.subframe_padding:self.subframe_padding+orig.shape[1]] = orig

        # Overlay right
        self.img_layout[self.subframe_padding:self.subframe_padding+segmented.shape[0], self.mid+self.subframe_padding:self.mid+self.subframe_padding+segmented.shape[1]] = segmented

    def execute(self):
        print("Loading model {}...".format(self.model_file))
        state = torch.load(self.model_file)

        print("Using model type: {}".format(self.model_type))
        print("In Channels: {}".format(self.in_channels))
        print("Out Channels: {}".format(self.out_channels))

        model = load_model_for_inference(
            self.in_channels,
            self.out_channels,
            self.model_type,
            self.device,
            state['state_dict']
        )

        print("Reading image {}...".format(self.input_img))
        img = cv2.imread(self.input_img)

        h, w, c = img.shape

        orig_img_width  = w
        orig_img_height = h

        self.draw_layout()

        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(img, self.dim) / 255
        input_img = input_img.transpose((2, 0, 1))

        x = torch.Tensor(
            np.array([input_img])
        ).to(
            self.device
        )

        result = model.forward(x)
        result = torch.argmax(result, 1).detach().cpu().numpy().astype(np.float32)

        result = (result.transpose((1, 2, 0)) / self.out_channels) * 255


        # Overlay on layout
        display_img_org = cv2.resize(img, self.display_dim)

        seg = cv2.resize(result, self.display_dim)

        display_img_seg = np.zeros((self.display_dim[0], self.display_dim[1], 3), np.uint8)
        display_img_seg[:,:,0] = seg
        display_img_seg[:,:,1] = seg
        display_img_seg[:,:,2] = seg


        self.overlay(display_img_org, display_img_seg)

        cv2.imshow("Frame", self.img_layout)

        cv2.waitKey()
