import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .VAE import vae_loss
from .classifier import *
from .loss import *


def fourier_time_embedding(t, num_frequencies=8):
    """Fourier embed scalar t into higher-dimensional space."""
    freqs = 2 ** torch.arange(num_frequencies, device=t.device) * math.pi
    angles = t * freqs  # (B, 1) * (F,) = (B, F)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, 2F)


def flow_matching_loss(flow_model, z0, z1, t):

    xt = (1 - t) * z0 + t * z1  # linear interpolation
    v_true = z1 - z0

    time_emb = fourier_time_embedding(t)

    v_pred = flow_model(xt, time_emb)

    loss = F.mse_loss(v_pred, v_true)

    return loss


def contrastive_loss_fn(arb_latent, clip_text_latent, temperature=0.07):

    arb_latent = F.normalize(arb_latent, dim=1)
    clip_text_latent = F.normalize(clip_text_latent, dim=1)

    logits = torch.matmul(arb_latent, clip_text_latent.T) / temperature # cosine similarities
    
    batch_size = logits.size(0)
    labels = torch.arange(batch_size, device=arb_latent.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    return (loss_i2t + loss_t2i) / 2


def intra_modality_supcon_loss(latents, labels, temperature=0.07):

    device = latents.device
    latents = F.normalize(latents, dim=1)
    
    sim_matrix = torch.matmul(latents, latents.T) / temperature  # cosine similarities

    self_mask = torch.eye(sim_matrix.size(0), device=device).bool()
    label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    positives_mask = label_mask & ~self_mask

    log_prob = F.log_softmax(sim_matrix, dim=1)
    loss = - (log_prob * positives_mask).sum(dim=1) / positives_mask.sum(dim=1).clamp(min=1)

    return loss.mean()
