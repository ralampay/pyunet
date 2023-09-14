import cv2
import numpy as np
import torch
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from lib.unet import UNet
from lib.unet_attn import UNetAttn
from lib.unet_attn_dp import UNetAttnDp
from lib.unet_attn_ghost import UNetAttnGhost
from lib.unet_attn_inverted_residual_block import UNetAttnInvertedResidualBlock
from lib.unet_attn_stacked_ghost_irb import UNetAttnStackedGhostIrb
from lib.depth.unet_depth import UNetDepth
from lib.depth.unet_attn_depth import UNetAttnDepth
from lib.depth.unet_attn_dp_depth import UNetAttnDpDepth

def rgb2mask(colors, image):
    rows, cols, _ = image.shape

    result = np.zeros((rows, cols))

    for y in range(0, rows):
        for x in range(0, cols):
            b, g, r = (image[y, x])

            for i in range(len(colors)):
                if colors[i] == [r, g, b]:
                    result[y, x] = i

    return result

def load_model_for_inference(in_channels, out_channels, model_type, device, state_dict):
    model = initialize_model(
        in_channels, 
        out_channels, 
        model_type, 
        device
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model

def initialize_model(in_channels, out_channels, model_type, device):
    model = None

    if model_type == 'unet':
        model = UNet(
            in_channels=in_channels,
            out_channels=out_channels
        )
    elif model_type == 'unet_attn':
        model = UNetAttn(
            in_channels=in_channels,
            out_channels=out_channels
        )
    elif model_type == 'unet_attn_dp':
        model = UNetAttnDp(
            in_channels=in_channels,
            out_channels=out_channels
        )
    elif model_type == 'unet_attn_ghost':
        model = UNetAttnGhost(
            in_channels=in_channels,
            out_channels=out_channels
        )
    elif model_type == 'unet_attn_inverted_residual_block':
        model = UNetAttnInvertedResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels
        )
    elif model_type == 'unet_attn_stacked_ghost_irb':
        model = UNetAttnStackedGhostIrb(
            in_channels=in_channels,
            out_channels=out_channels
        )

    # Depth Models
    elif model_type == 'unet_depth':
        model = UNetDepth(
            in_channels=in_channels,
            out_channels=out_channels
        )
    elif model_type == 'unet_attn_depth':
        model = UNetAttnDepth(
            in_channels=in_channels,
            out_channels=out_channels
        )
    elif model_type == 'unet_attn_dp_depth':
        model = UNetAttnDpDepth(
            in_channels=in_channels,
            out_channels=out_channels
        )

    else:
        raise ValueError(f'Unsupported model_type {model_type}')

    return model.to(device)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def dice_score(pred, true, k=0):
    intersection = np.sum(pred[true==k]) * 2.0
    dice = intersection / (np.sum(pred) + np.sum(true))

    return dice

def get_image(file, dim):
    img = cv2.resize(
        cv2.cvtColor(
            cv2.imread(file),
            cv2.COLOR_BGR2RGB 
        ),
        dim
    ) / 255

    return img

def get_mask(file, dim):
    img = cv2.resize(
        cv2.imread(
            file,
            0
        ),
        dim
    )

    return img

def get_predicted_img(img, model, device='cpu', out_channels=4):
    model.to(device)

    input_img = img.transpose((2, 0, 1))

    x = torch.Tensor(np.array([input_img])).to(device)

    result = model.forward(x)
    result = torch.argmax(result, 1).detach().cpu().numpy().astype(np.int32)
    result = result.transpose((1, 2, 0))

    return result
