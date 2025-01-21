import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalSRLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
        
    def reconstruction_loss(self, pred, target):
        return F.l1_loss(pred, target)
        
    def kl_loss(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
    def counterfactual_loss(self, pred_cf, target_cf):
        return F.mse_loss(pred_cf, target_cf)
        
    def intervention_loss(self, z_int, z, xi):
        # Regularize intervention strength
        strength_reg = torch.mean(xi.pow(2))
        # Encourage sparse interventions
        sparsity_reg = torch.mean(torch.abs(z_int - z))
        return strength_reg + self.weights['sparsity'] * sparsity_reg
        
    def forward(self, outputs, targets):
        losses = {}
        
        # Reconstruction loss
        losses['rec'] = self.reconstruction_loss(outputs['sr'], targets['hr'])
        
        # KL divergence
        losses['kl'] = self.kl_loss(outputs['mu'], outputs['logvar'])
        
        # Counterfactual loss
        losses['cf'] = self.counterfactual_loss(outputs['sr_cf'], targets['hr_cf'])
        
        # Intervention loss
        losses['int'] = self.intervention_loss(outputs['z_int'], outputs['z'], outputs['xi'])
        
        # Total loss
        total_loss = sum(w * losses[k] for k, w in self.weights.items())
        
        return total_loss, losses