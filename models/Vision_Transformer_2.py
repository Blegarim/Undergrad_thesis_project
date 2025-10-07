import torch, torch.nn as nn, torch.nn.functional as F

class GEGLU(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff * 2)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.fc1(x)
        x1, x2 = x.chunk(2, dim=-1) 
        return self.dropout(self.fc2(F.gelu(x1) * x2))
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model=128, num_heads=8, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = GEGLU(d_model, dim_feedforward, dropout)

    def forward(self, x, key_padding_mask=None):
        y = self.norm1(x)
        attn_output, _ = self.self_attn(y, y, y, key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_output)

        y = self.norm2(x)
        ffn_output = self.ffn(y)
        x = x +self.dropout(ffn_output)

        return x
    
class ViT_Hierarchical(nn.Module):
    def __init__(
            self,
            img_size=128,
            patch_size=16,
            in_channels=3
    )