import torch
import torch.nn.functional as F
import random
import numpy as np

def gaussian_blur(x, kernel_size, sigma):
    channels = x.shape[1]
    
    # Create Gaussian kernel
    kernel = torch.zeros((channels, 1, kernel_size, kernel_size))
    center = kernel_size // 2
    
    for c in range(channels):
        for i in range(kernel_size):
            for j in range(kernel_size):
                diff = ((i-center)**2 + (j-center)**2) / (2.0 * sigma**2)
                kernel[c,0,i,j] = np.exp(-diff) / (2 * np.pi * sigma**2)
                
    kernel = kernel / kernel.sum(dim=(2,3), keepdim=True)
    kernel = kernel.to(x.device)
    
    # Apply blur
    return F.conv2d(x.unsqueeze(0), kernel, padding=center)[0]

def add_noise(x, noise_std):
    noise = torch.randn_like(x) * noise_std
    return torch.clamp(x + noise, 0, 1)

def jpeg_compression(x, quality):
    # Simplified JPEG compression simulation
    # Convert to YCbCr
    rgb_to_ycbcr = torch.tensor([
        [0.299, 0.587, 0.114],
        [-0.169, -0.331, 0.5],
        [0.5, -0.419, -0.081]
    ]).to(x.device)
    
    ycbcr = torch.matmul(x.permute(1,2,0), rgb_to_ycbcr.T)
    
    # Quantization
    scale = quality / 100.0
    quant_factor = torch.ceil(ycbcr / scale) * scale
    
    # Back to RGB
    ycbcr_to_rgb = torch.inverse(rgb_to_ycbcr)
    rgb = torch.matmul(quant_factor, ycbcr_to_rgb.T)
    
    return rgb.permute(2,0,1).clamp(0, 1)

def random_degradation(x, scale, params=None):
    """Apply random degradation to image"""
    if params is None:
        params = {
            'kernel_size': random.choice([3, 5, 7]),
            'sigma': random.uniform(0.1, 3.0),
            'noise_std': random.uniform(0, 0.1),
            'jpeg_quality': random.randint(30, 95)
        }
    
    # Apply degradations
    x = gaussian_blur(x, params['kernel_size'], params['sigma'])
    x = add_noise(x, params['noise_std'])
    x = jpeg_compression(x, params['jpeg_quality'])
    
    # Downsample
    x = F.interpolate(x.unsqueeze(0), scale_factor=1/scale, mode='bicubic')[0]
    
    return x, params