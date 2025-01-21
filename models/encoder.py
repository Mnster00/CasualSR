import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return self.relu(out)

class Encoder(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=8):
        super().__init__()
        
        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1)
        
        # Downsampling layers
        self.down1 = nn.Conv2d(num_channels, num_channels*2, kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Conv2d(num_channels*2, num_channels*4, kernel_size=4, stride=2, padding=1)
        
        # Residual blocks
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResBlock(num_channels*4))
        self.res_blocks = nn.Sequential(*blocks)
        
        # Final convolution
        self.final_conv = nn.Conv2d(num_channels*4, num_channels*8, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Initial features
        x = self.relu(self.init_conv(x))
        
        # Downsample
        x = self.relu(self.down1(x))
        x = self.relu(self.down2(x))
        
        # Residual blocks
        x = self.res_blocks(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x