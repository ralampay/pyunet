import sys
import argparse
import os
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from modules.train import Train
from modules.forward import Forward
from modules.monitor import Monitor

mode_choices = [
    "train",
    "forward",
    "monitor"
]

def main():
    parser = argparse.ArgumentParser(description="PyUNET: Python implementation of UNET")

    parser.add_argument("--mode", help="Mode to be used", choices=mode_choices, type=str, required=True)
    parser.add_argument("--img-width", help="Image width", type=int, default=260)
    parser.add_argument("--img-height", help="Image height", type=int, default=260)
    parser.add_argument("--device", help="Device used for training", choices=["cpu", "cuda"], type=str, default="cpu")
    parser.add_argument("--gpu-index", help="GPU index", type=int, default=0)
    parser.add_argument("--input-img-dir", help="Input image directory", type=str)
    parser.add_argument("--input-mask-dir", help="Input mask directory", type=str)
    parser.add_argument("--epochs", help="Epoch count", type=int, default=100)
    parser.add_argument("--learning-rate", help="Learning rate", type=float, default=0.001)
    parser.add_argument("--model-file", help="Model file", type=str, default="model.pth")
    parser.add_argument("--batch-size", help="Batch size", type=int, default=1)
    parser.add_argument("--input-img", help="Input image", type=str, required=False)
    parser.add_argument("--in-channels", help="In Channels", type=int, default=3)
    parser.add_argument("--out-channels", help="Out Channels", type=int, default=2)
    parser.add_argument("--features", help="Features", type=int, nargs='+', default=[64, 128, 256, 512])
    parser.add_argument("--video", help="Video index", type=str, default="0")

    args = parser.parse_args()

    mode            = args.mode
    img_width       = args.img_width
    img_height      = args.img_height
    device          = args.device
    gpu_index       = args.gpu_index
    input_img_dir   = args.input_img_dir
    input_mask_dir  = args.input_mask_dir
    epochs          = args.epochs
    learning_rate   = args.learning_rate
    model_file      = args.model_file
    batch_size      = args.batch_size
    input_img       = args.input_img
    in_channels     = args.in_channels
    out_channels    = args.out_channels
    features        = args.features
    video     = args.video

    if mode =="train":
        params = {
            'img_width':        img_width,
            'img_height':       img_height,
            'device':           device,
            'gpu_index':        gpu_index,
            'input_img_dir':    input_img_dir,
            'input_mask_dir':   input_mask_dir,
            'epochs':           epochs,
            'learning_rate':    learning_rate,
            'model_file':       model_file,
            'batch_size':       batch_size,
            'in_channels':      in_channels,
            'out_channels':     out_channels,
            'features':         features
        }

        cmd = Train(params=params)
        cmd.execute()
    elif mode =="forward":
        params = {
            'model_file':   model_file,
            'input_img':    input_img,
            'gpu_index':    gpu_index,
            'device':       device
        }

        cmd = Forward(params=params)
        cmd.execute()
    elif mode =="monitor":
        params = {
            'model_file':   model_file,
            'video':        video,
            'gpu_index':    gpu_index,
            'device':       device
        }

        cmd = Monitor(params=params)
        cmd.execute()
    else:
        raise ValueError("Invalid mode {}".format(mode))

if __name__ == '__main__':
    main()
