import sys
import os
import torch
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.unet import UNet

class Monitor:
    def __init__(self, params={}):
        self.params = params

        self.device     = params.get('device')
        self.gpu_index  = params.get('gpu_index')
        self.video      = params.get('video')
        self.model_file = params.get('model_file')

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

        saved_params = state['params']

        img_width   = saved_params.get('img_width')
        img_height  = saved_params.get('img_height')
        dim         = (img_width, img_height)

        in_channels     = saved_params.get('in_channels') or 3
        out_channels    = saved_params.get('out_channels') or 3
        features        = saved_params.get('features') or [64, 128, 256, 512]

        model   = UNet(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    features=features
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

                image = (cv2.resize(frame, dim) / 255).transpose((2, 0, 1))
                tensor_image = torch.Tensor(image)

                x = torch.tensor([tensor_image.numpy()]).to(self.device)

                result = cv2.resize(model.forward(x).detach().cpu().numpy()[0].transpose(1, 2, 0), original_dim)

                cv2.imshow("Original", frame)
                cv2.imshow("Result", result)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            else:
                break

        cap.release()
        
        cv2.destroyAllWindows()
