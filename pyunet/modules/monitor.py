import sys
import os
import torch
import cv2
import time
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils import load_model_for_inference

COLOR_RED = np.array([0, 0, 255], dtype='uint8')

class Monitor:
    def __init__(self, params={}):
        self.params = params

        self.device         = params.get('device') or 'cuda'
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

        self.colors = []

        for i in range(self.out_channels - 1):
            self.colors.append(np.array([
                int(np.random.uniform(0, 255)), 
                int(np.random.uniform(0, 255)), 
                int(np.random.uniform(0, 255))
            ], dtype='uint8'))

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
        state = torch.load(self.model_file, map_location=self.device)

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

        cap = cv2.VideoCapture(self.video)

        if cap.isOpened() == False:
            print("Error in opening video {}".format(self.video))
            sys.exit()

        # used to record the time at which we processed last frame
        prev_frame_time = 0

        # used to record the time at which we processed current frame
        new_frame_time = 0

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

                display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                result_display_frame = cv2.resize(display_frame, self.display_dim)
                result_segmented = cv2.resize(result, self.display_dim)
                result_segmented = np.where(result_segmented != 0, 1.0, 0.0).astype('uint8')

                contours, hierarchy = cv2.findContours(result_segmented, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                # find the biggest countour (c) by the area
                #c = max(contours, key = cv2.contourArea)
                #x,y,w,h = cv2.boundingRect(c)

                # draw the biggest contour (c) in green
                #cv2.rectangle(result_display_frame,(x,y),(x+w,y+h),(255,0,0),2)

                masked_img = np.where(result_segmented[...,None], COLOR_RED, result_display_frame)
                result_mask_overlay = cv2.addWeighted(result_display_frame, 0.5, masked_img, 0.8, 0)
                cv2.drawContours(result_mask_overlay, contours, -1, (0, 255, 0), 3)

                cv2.imshow("Result", result_mask_overlay)

                new_frame_time = time.time()

                fps = round(1 / (new_frame_time - prev_frame_time), 3)
                prev_frame_time = new_frame_time

                print(f'\rFPS: {fps}', end="")

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            else:
                break

        print("")
        cap.release()
        
        cv2.destroyAllWindows()
