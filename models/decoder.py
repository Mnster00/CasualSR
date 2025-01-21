import torch
import torch.nn as nn

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 
                                    kernel_size=4, stride=2, padding=1)
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.up(x))
        x = self.relu(self.conv(x))
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim, num_channels=64, scale_factor=4):
        super().__init__()
        
        self.init_conv = nn.Conv2d(latent_dim, num_channels*8, kernel_size=3, padding=1)
        
        # Upsampling layers
        self.up1 = UpBlock(num_channels*8, num_channels*4)
        self.up2 = UpBlock(num_channels*4, num_channels*2)
        
        # Additional upsampling for different scale factors
        up_blocks = []
        curr_scale = 2
        while curr_scale < scale_factor:
            up_blocks.extend([
                UpBlock(num_channels*2, num_channels),
                nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            ])
            curr_scale *= 2
        self.up_blocks = nn.Sequential(*up_blocks)
        
        # Final convolution
        self.final_conv = nn.Conv2d(num_channels, 3, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.init_conv(x))
        x = self.up1(x)
        x = self.up2(x)
        x = self.up_blocks(x)
        x = self.final_conv(x)
        return x