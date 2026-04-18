import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, mult: float = 1.5):
        super().__init__()
        hidden_dim = int(dim * mult)
        self.block = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.block(x) + x)


class ResNet(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256, 
        num_blocks: int = 4,
        block_mult: float = 2 
    ):
        super().__init__()
        self.input_layer = nn.Linear(16 + latent_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            ResidualBlock(dim=hidden_dim, mult=block_mult) for _ in range(num_blocks)
        ])

        self.output_layer = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, latent_dim]
        # t: [batch_size, 16]
        xt = torch.cat([x, t], dim=1)

        h = F.silu(self.input_layer(xt))

        for block in self.blocks:
            h = block(h)

        v = self.output_layer(h)
        return v