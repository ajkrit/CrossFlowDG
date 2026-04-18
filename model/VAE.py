import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Type

class VAE(nn.Module):
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 hidden_dims: List[int] = [512, 256],
                 activation_fn: Type[nn.Module] = nn.GELU,
                 use_batchnorm: bool = True,
                 dropout_rate: float = 0.0):
        super().__init__()
        self.latent_dim = latent_dim

        # --- Encoder ---
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            if use_batchnorm:
                encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(activation_fn())
            if dropout_rate > 0:
                encoder_layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_log_var = nn.Linear(hidden_dims[-1], latent_dim)

        # --- Decoder ---
        decoder_layers = []
        reversed_hidden_dims = list(reversed(hidden_dims))
        in_dim = latent_dim
        for h_dim in reversed_hidden_dims:
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            if use_batchnorm:
                decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(activation_fn())
            if dropout_rate > 0:
                decoder_layers.append(nn.Dropout(dropout_rate))
            in_dim = h_dim
        
        self.decoder_net = nn.Sequential(*decoder_layers)
        self.final_layer = nn.Linear(reversed_hidden_dims[-1], input_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_net(z)
        return self.final_layer(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z
        
def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # MSE loss for reconstruction
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    
    # KL divergence for regularization
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kld
