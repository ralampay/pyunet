import sys
import os
import torch
import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.unet import UNet
from lib.unet_rd import UNetRd

class Monitor:
    def __init__(self, params={}):
        self.params = params

        self.device         = params.get('device')
        self.gpu_index      = params.get('gpu_index')
        self.video          = params.get('video')
        self.model_file     = params.get('model_file')
        self.img_width      = params.get('img_width')
        self.img_height     = params.get('img_height')
        self.display_width  = params.get('display_width') or 800
        self.display_height = params.get('display_height') or 640
        self.in_channels    = params.get('in_channels') or 3
        self.out_channels   = params.get('out_channels') or 2
        self.model_type     = params.get('model_type') or 'unet'
        self.dim            = (self.img_width, self.img_height)
        self.display_dim    = (self.display_width, self.display_height)

        print("Dimension: {}".format(self.dim))

    def execute(self):
        try:
            self.video = int(self.video)
        except ValueError:
            print("video {} not an index. Treating as file...".format(self.video))

        print("Starting monitor mode on video index {}...".format(self.video))

        if self.device == 'cuda':
            print("CUDA Device: {}".format(torch.cuda.get_device_name(self.gpu_index)))
            self.device = "cuda:{}".format(self.gpu_index)

        print("Loading model {}...".format(self.model_file))
        state = torch.load(self.model_file)

        print("Using model type: {}".format(self.model_type))
        print("In Channels: {}".format(self.in_channels))
        print("Out Channels: {}".format(self.out_channels))

        if self.model_type == 'unet':
            model = UNet(
                in_channels=self.in_channels,
                out_channels=self.out_channels
            ).to(self.device)
        elif self.model_type == 'unet_rd':
            model = UNetRd(
                in_channels=self.in_channels,
                out_channels=self.out_channels
            ).to(self.device)

        model.load_state_dict(state['state_dict'])

        cap = cv2.VideoCapture(self.video)

        if cap.isOpened() == False:
            print("Error in opening video {}".format(self.video))
            sys.exit()

        while(cap.isOpened()):
            ret, frame = cap.read()
        
            if ret == True:
                rows, cols, _   = frame.shape
                original_dim    = (cols, rows)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                image = cv2.resize(frame, self.dim) / 255
                image = image.transpose((2, 0, 1))

                x = torch.Tensor(
                    np.array([image])
                ).to(
                    self.device
                )

                result = model.forward(x)
                result = torch.argmax(result, 1).detach().cpu().numpy().astype(np.float32)

                result = result.transpose((1, 2, 0)) / self.out_channels

                cv2.imshow("Original", cv2.resize(frame, self.display_dim))
                cv2.imshow("Segmented", cv2.resize(result, self.display_dim))

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            else:
                break

        cap.release()
        
        cv2.destroyAllWindows()
