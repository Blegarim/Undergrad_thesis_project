import torch
import torch.nn as nn

class EnsembleModel(nn.Module):
    def __init__(self, motion_enc, vit, cross_attention, d_model=128):
        super().__init__()
        self.motion_enc = motion_enc
        self.vit = vit
        self.cross_attention = cross_attention
        self.image_norm = nn.LayerNorm(d_model)
        self.motion_norm = nn.LayerNorm(d_model)

    def forward(self, images_tight, images_context, motions, return_feats=False):
        # --- Vision Transformer branch ---
        image_feats = self.vit(images_context)        # [B, T, D]
        image_feats = self.image_norm(image_feats)

        # --- Motion branch ---
        motion_out = self.motion_enc(motions, images_tight)     # [B, T, D]
        motion_feats = self.motion_norm(motion_out)


        # --- Cross-attention fusion ---
        logits = self.cross_attention(motion_feats, image_feats)  # dict of logits per task

        if return_feats:
            return logits, image_feats, motion_feats

        return logits

    
