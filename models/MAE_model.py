import torch
import torch.nn as nn
import torch.nn.functional as F

class ViT_Decoder(nn.Module):
    def __init__(self, d_model=128, depth=8, num_heads=8, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)
