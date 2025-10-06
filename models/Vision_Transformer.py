import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout = dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = GEGLU(d_model, dim_feedforward, dropout)
    
    def forward(self, x, key_padding_mask=None):
        """
        x: Tensor of shape [batch_size, seq_len, d_model]
        mask: Optional mask tensor of shape [batch_size, 1, seq_len]
        """
        # Self-attention
        y = self.norm1(x)
        attn_output, _ = self.self_attn(y, y, y, key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_output)

        y = self.norm2(x)
        ffn_output = self.ffn(y)
        x = x + self.dropout(ffn_output)

        return x

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_channels=3, embedding_dim=128):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x shape: [B, T, C(=3), H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W) # Merge batch and time: [B*T, C, H, W]
        x = self.projection(x)    # Shape: [B*T, D, H/P, W/P]
        x = x.flatten(2)         # Shape: [B*T, D, N], where N = (H/P)*(W/P)
        x = x.transpose(1, 2)    # Shape: [B*T, N, D]
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_channels=3, embedding_dim=128,
                 num_heads=8, num_layers=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embedding_dim)
        num_patches = (img_size // patch_size) * (img_size // patch_size)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embedding_dim))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.encoder_layer = nn.ModuleList([
            TransformerEncoderBlock(d_model=embedding_dim, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = self.patch_embedding(x) # Shape: [B*T, N, D]
        N = x.size(1) # Number of patches

        cls_tokens = self.cls_token.expand(B * T, -1, -1) # Shape: [B*T, 1, D]
        x = torch.cat((cls_tokens, x), dim=1) # Shape: [B*T, N+1, D]
        x = x + self.pos_embedding[:, :N + 1, :] # Add positional embedding
        x = self.dropout(x)

        for layer in self.encoder_layer:
            x = layer(x)
        
        x = self.norm(x)
        cls_output = x[:, 0] # Extract CLS token output
        return cls_output.view(B, T, -1) # Shape: [B, T, D]
    
if __name__ == '__main__':
    dummy_input = torch.randn(8, 10, 3, 128, 128)  # [B, T, C, H, W]
    model = VisionTransformer(img_size=128, patch_size=16, in_channels=3, embedding_dim=128, num_heads=4, num_layers=2)
    output = model(dummy_input)
    print(output.shape)  # Expected: [8, 10, 128]