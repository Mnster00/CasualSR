import torch
import torch.nn as nn

class AdaptiveIntervention(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        
        # Intervention strength prediction
        self.strength_net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            nn.Linear(latent_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Intervention generation
        self.intervention_net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh()
        )
        
        # Sparse intervention mask
        self.mask_net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid()
        )
        
    def forward(self, z, c=None):
        # Predict intervention strength
        xi = self.strength_net(z)
        
        # Generate intervention
        delta_z = self.intervention_net(z)
        
        # Generate sparse mask
        mask = self.mask_net(z)
        
        # Apply intervention
        z_int = z + xi * mask * delta_z
        
        # Optional context conditioning
        if c is not None:
            z_int = z_int + c
            
        return z_int, xi, mask