import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model=128, num_heads=8, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout = dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        """
        x: Tensor of shape [batch_size, seq_len, d_model]
        mask: Optional mask tensor of shape [batch_size, 1, seq_len]
        """
        # Self-attention
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))

        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x
    
class MotionTransformer(nn.Module):
    def __init__(self, d_model=128, max_len=100, num_heads=8, num_layers=2, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(3, d_model)  # Input: [batch_size, seq_len, 3] → Output: [batch_size, seq_len, d_model]
        self.positional_encoding = nn.Embedding(max_len, d_model) # Positional encoding for sequence length

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model=d_model, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.pooling = nn.AdaptiveAvgPool1d(1) # Pooling layer to aggregate features

    def forward(self, motions):
        """
        motions: Tensor of shape [batch_size, seq_len, 3] (cx, cy, dt)
        """
        B, T, _ = motions.shape
        T = min(T, self.positional_encoding.num_embeddings)
        x = self.input_proj(motions) # Shape: [batch_size, seq_len, d_model]

        positions = torch.arange(T, device=motions.device).unsqueeze(0).expand(B, T) # Shape: [batch_size, seq_len]
        pos_embed = self.positional_encoding(positions) # Shape: [batch_size, seq_len, d_model]
        x = x + pos_embed # Add positional encoding

        x = x.permute(1, 0, 2) # Shape: [seq_len, batch_size, d_model] for nn.MultiheadAttention

        for layer in self.encoder_layers:
            x = layer(x)

        x = x.permute(1, 0, 2) # Shape: [batch_size, seq_len, d_model]

        return x
    
def test_motion_transformer():
    batch_size = 4
    seq_len = 20
    input_dim = 3
    d_model = 128

    motions = torch.randn(batch_size, seq_len, input_dim)
    model = MotionTransformer(d_model=d_model, max_len=100, num_heads=8, num_layers=2)
    
    out = model(motions)

    print("Input shape:", motions.shape)         # Should be [4, 20, 3]
    print("Output shape:", out.shape)            # Should be [4, 128]
    print("Output sample:", out[0, :5])          # First 5 values from first sample

if __name__ == "__main__":
    test_motion_transformer()
