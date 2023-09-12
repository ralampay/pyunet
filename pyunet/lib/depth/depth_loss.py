import torch
import torch.nn as nn
import pytorch_msssim

def depth_loss(predictions, depthmap, weight_pointwise=0.5, weight_gradient=0.25, weight_ssim=0.25):
    mse_loss = nn.MSELoss()

    # Compute pointwise loss
    loss_pointwise = mse_loss(predictions, depthmap)

    # Gradient loss

    # Define custom gradient kernels for x and y directions
    gradient_kernel_x = torch.tensor([[0, 0, 0],
                                      [-1, 0, 1],
                                      [0, 0, 0]], dtype=torch.float32).to(predictions.device).view(1, 1, 3, 3)

    gradient_kernel_y = torch.tensor([[0, -1, 0],
                                      [0, 0, 0],
                                      [0, 1, 0]], dtype=torch.float32).to(predictions.device).view(1, 1, 3, 3)

    # Calculate gradients along the x-axis
    gradient_predictions_x = nn.functional.conv2d(predictions, gradient_kernel_x, padding=1)
    gradient_depthmap_x = nn.functional.conv2d(depthmap, gradient_kernel_x, padding=1)

    # Calculate gradients along the y-axis
    gradient_predictions_y = nn.functional.conv2d(predictions, gradient_kernel_y, padding=1)
    gradient_depthmap_y = nn.functional.conv2d(depthmap, gradient_kernel_y, padding=1)

    # Calculate the squared differences for x and y gradients
    squared_diff_x = (gradient_predictions_x - gradient_depthmap_x) ** 2
    squared_diff_y = (gradient_predictions_y - gradient_depthmap_y) ** 2

    # Compute the mean of the squared differences for x and y gradients
    gradient_loss_x = torch.mean(squared_diff_x)
    gradient_loss_y = torch.mean(squared_diff_y)

    g_loss = gradient_loss_x + gradient_loss_y

    # SSIM loss
    ssim_loss = 1.0 - pytorch_msssim.ssim(predictions, depthmap)

    return weight_pointwise * loss_pointwise + weight_gradient * g_loss + weight_ssim * ssim_loss
