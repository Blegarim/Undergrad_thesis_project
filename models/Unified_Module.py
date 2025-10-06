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
        # Extract CNN features per frame sequence
        image_feats = self.norm(self.tcngru(images)) # Shape: [batch_size, seq_len, d_model]

        motion_out = self.vit(motions)

        # Extract motion features
        motion_feats, motion_cls = self.norm(motion_out)

        # Cross-attention between image features and motion features
        logits = self.cross_attention(motion_feats, image_feats) # Shape: [batch_size, num_classes]
        if return_feats:
            return logits, image_feats, motion_feats
        return logits
    
