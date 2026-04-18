import torch
import torch.nn as nn
import torch.nn.functional as F
from .VAE import vae_loss
from .classifier import *
from .loss import *

EPS = 1e-8

class CrossFlowDG(nn.Module):
    def __init__(self, clip_model=None, txt_vae=None, encoder=None, img_vae=None, flow_model=None, classifier=None, device="cuda"):
        super().__init__()
        self.device = device


        if clip_model is not None:
            self.clip_model = clip_model.to(self.device)
            for param in self.clip_model.parameters():
                param.requires_grad = False
        if txt_vae is not None:
            self.txt_vae = txt_vae.to(self.device)
        self.encoder = encoder.to(self.device)
        self.img_vae = img_vae.to(self.device)
        self.flow_model = flow_model.to(self.device)
        self.classifier = classifier.to(self.device)


    def forward(self, images, input_ids=None, attention_mask=None, ctargets=None, num_steps=1, epoch=None):

        images = images.to(self.device)
        if ctargets is not None:
            ctargets = ctargets.to(self.device)
        if input_ids is not None:
            input_ids = input_ids.squeeze(1).to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(1).to(self.device)

        loss = {}

        img_enc = self.encoder(images)


        pooled_feats = []
        for fmap in img_enc:
            # Global average pool to shape [batch_size, channels, 1, 1]
            pooled = F.adaptive_avg_pool2d(fmap, 1)
            # Flatten last two dims to get [batch_size, channels]
            pooled = pooled.view(pooled.size(0), -1)
            pooled_feats.append(pooled)

        # Concatenate all pooled features along channel dimension
        img_emb = torch.cat(pooled_feats, dim=1)

        if self.training:
            img_emb_rec, img_mu, img_logvar, img_latent = self.img_vae(img_emb)
            loss['img_vae'] = vae_loss(img_emb_rec, img_emb, img_mu, img_logvar)
        else:
            img_mu, img_logvar = self.img_vae.encode(img_emb)
            img_latent = img_mu

            # clip_text_emb = self.clip_model.text_model(
            #     input_ids=input_ids, 
            #     attention_mask=attention_mask).pooler_output

            # clip_text_mu, _ = self.txt_vae.encode(clip_text_emb)
            # clip_text_latent = clip_text_mu

        if self.training:
            # Sample random t for flow matching
            t1 = 1.0 - EPS
            T = img_latent.size(0) - 2
            t = torch.linspace(0, t1, T + 1, device=img_latent.device)
            t = t + torch.rand_like(t) * t1 / T
            t[-1] = t1 - EPS
            t = F.pad(t, (1, 0), value=0.0 + EPS)
            t = t.unsqueeze(1)  # (B, 1)
            time_emb = fourier_time_embedding(t)  # (B, 16)

            
            clip_text_emb = self.clip_model.text_model(
                input_ids=input_ids, 
                attention_mask=attention_mask).pooler_output


            clip_text_emb_rec, clip_mu, clip_logvar, clip_text_latent = self.txt_vae(clip_text_emb)
            loss['txt_vae'] = vae_loss(clip_text_emb_rec, clip_text_emb, clip_mu, clip_logvar)

            loss['contrastive'] = (contrastive_loss_fn(img_latent, clip_text_latent)
                                + 1/2*(intra_modality_supcon_loss(img_latent, ctargets) 
                                + intra_modality_supcon_loss(clip_text_latent, ctargets)))


            loss['flow_matching'] = flow_matching_loss(
                self.flow_model,
                z0=img_latent,
                z1=clip_text_latent,
                t=t
            )


            x = img_latent
            dt = 1.0 / num_steps

            for i in range(num_steps):
                t = i * dt * torch.ones(x.size(0), 1, device=x.device)
                time_emb = fourier_time_embedding(t)
                
                v_pred = self.flow_model(x, time_emb)
                x = x + v_pred * dt

            logits = self.classifier(x) if self.classifier else None

        else:

            x = img_latent
            dt = 1.0 / num_steps

            # xs = [x.detach()] # uncomment to store valid/target trajectories

            for i in range(num_steps):
                t = i * dt * torch.ones(x.size(0), 1, device=x.device)
                time_emb = fourier_time_embedding(t)
                
                v_pred = self.flow_model(x, time_emb)
                x = x + v_pred * dt

                # xs.append(x.detach())

            logits = self.classifier(x) if self.classifier else None

        if self.training:
            loss['classifier'] = F.cross_entropy(logits, ctargets)
            return logits, loss, img_latent, clip_text_latent
        else:
            return logits, None, None, None
            # return logits, xs, img_latent, clip_text_latent

