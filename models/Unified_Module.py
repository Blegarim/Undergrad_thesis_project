import torch
import torch.nn as nn

class EnsembleModel(nn.Module):
    def __init__(self, tcngru, vit, cross_attention, d_model=128):
        super().__init__()
        self.tcngru = tcngru
        self.vit = vit
        self.cross_attention = cross_attention
        self.norm = nn.LayerNorm(d_model)

    def forward(self, images, motions, return_feats=False):
        # --- Vision Transformer branch ---
        image_feats = self.vit(images)        # [B, T, D]
        image_feats = self.norm(image_feats)

        # --- Motion branch ---
        motion_out = self.tcngru(motions)     # [B, T, D]
        motion_feats = self.norm(motion_out)


        # --- Cross-attention fusion ---
        logits = self.cross_attention(motion_feats, image_feats)  # dict of logits per task

        if return_feats:
            return logits, image_feats, motion_feats
        return logits

    
