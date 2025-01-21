import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class StructuredPrior(nn.Module):
    def __init__(self, graph_size, mixture_components=4):
        super().__init__()
        self.graph_size = graph_size
        self.K = mixture_components
        
        # Mixture weights
        self.pi = nn.Parameter(torch.ones(mixture_components) / mixture_components)
        # Edge probabilities for each component
        self.p = nn.Parameter(torch.rand(mixture_components))
        
    def forward(self):
        # Generate adjacency matrix from mixture
        pi = F.softmax(self.pi, dim=0)
        p = torch.sigmoid(self.p)
        
        adj = torch.zeros(self.graph_size, self.graph_size)
        for k in range(self.K):
            mask = torch.bernoulli(torch.ones(self.graph_size, self.graph_size) * p[k])
            adj += pi[k] * mask
            
        return torch.sigmoid(adj)

class CausalModule(nn.Module):
    def __init__(self, input_dim=512, latent_dim=256, graph_size=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.graph_size = graph_size
        
        # Variational encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim * 2)
        )
        
        # Graph structure
        self.graph_prior = StructuredPrior(graph_size)
        
        # GNN layers for causal propagation
        self.gnn_layers = nn.ModuleList([
            nn.Linear(latent_dim, latent_dim) for _ in range(3)
        ])
        
        # Intervention network
        self.intervention = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh()
        )
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Encode to latent space
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)
        
        # Reshape to graph structure
        z = z.view(batch_size, self.graph_size, -1)
        
        # Get adjacency matrix
        adj = self.graph_prior()
        
        # Causal propagation through GNN
        for gnn in self.gnn_layers:
            z_msg = torch.bmm(adj.expand(batch_size, -1, -1), z)
            z = gnn(z_msg) + z  # Skip connection
            
        # Generate intervention
        z_int = self.intervention(z.view(batch_size, -1))
        
        return z_int, mu, logvar
