import torch
import torch.nn as nn

class MultimodalModel(nn.Module):
    def __init__(self, cnn_backbone, motion_transformer, cross_attention, d_model=128):
        super().__init__()
        self.cnn_backbone = cnn_backbone
        self.motion_transformer = motion_transformer
        self.cross_attention = cross_attention
        self.norm = nn.LayerNorm(d_model)

    def forward(self, images, motions, return_feats=False):
        # Extract CNN features per frame sequence
        image_feats = self.norm(self.cnn_backbone(images)) # Shape: [batch_size, seq_len, d_model]

        # Extract motion features
        motion_feats = self.norm(self.motion_transformer(motions))

        # Cross-attention between image features and motion features
        logits = self.cross_attention(motion_feats, image_feats) # Shape: [batch_size, num_classes]
        if return_feats:
            return logits, image_feats, motion_feats
        return logits
    
