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
    
class MotionTransformer(nn.Module):
    def __init__(self, d_model=128, max_len=100, 
                 num_heads=8, num_layers=2, 
                 dim_feedforward=512, dropout=0.1,
                 conv_k=3, conv_out=128):
        '''
        d_model: Dimension of the model
        max_len: Maximum length of the input sequence
        num_heads: Number of attention heads
        num_layers: Number of transformer encoder layers
        dim_feedforward: Dimension of the feedforward network
        dropout: Dropout rate
        conv_k: Kernel size for convolutional layer
        conv_out: Output channels for convolutional layer
        '''
        super().__init__()
        # pre_conv for motion smoothing: input C=3 (cx, cy, dt) to conv_out
        conv_out = d_model  # Ensure conv_out matches d_model for consistency
        self.pre_conv = nn.Conv1d(in_channels=3, out_channels=conv_out, kernel_size=conv_k, padding=(conv_k - 1) // 2, bias=True)
        self.pre_norm = nn.LayerNorm(conv_out)
        self.pre_activation = nn.GELU()

        self.pos_embedding = nn.Embedding(max_len+1, d_model) # Positional encoding for CLS token

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(d_model=d_model, num_heads=num_heads, dim_feedforward=dim_feedforward, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, motions, key_padding_mask=None):
        """
        motions: Tensor of shape [batch_size, seq_len, 3] (cx, cy, dt)
        """
        B, T, C = motions.shape 
        assert C == 3 # Ensure input has 3 channels

        x = motions.permute(0, 2, 1) # Shape: [batch_size, 3, seq_len]
        x = self.pre_conv(x) # Shape: [batch_size, conv_out==d_model, seq_len]
        x = x.permute(0, 2, 1) #Shape: [batch_size, seq_len, d_model]   
        x = self.pre_activation(x)
        x = self.pre_norm(x)

        #Append CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # Shape: [batch_size, seq_len+1, d_model]
        seq_len = x.shape[1]
        assert seq_len <= self.pos_embedding.num_embeddings, f"Sequence length {seq_len} exceeds maximum {self.pos_embedding.num_embeddings}, increase max_len."

        # Update mask to account for CLS (never masked)
        if key_padding_mask is not None:
            cls_pad = torch.zeros((B, 1), dtype=torch.bool, device=motions.device)
            key_padding_mask = torch.cat((cls_pad, key_padding_mask), dim=1)

        # Add positional encoding
        positions = torch.arange(seq_len, device=motions.device).unsqueeze(0).expand(B, seq_len) # Shape: [batch_size, seq_len]
        pos_embed = self.pos_embedding(positions) # Shape: [batch_size, seq_len, d_model]
        x = x + pos_embed # Add positional encoding

        for layer in self.encoder_layers:
            x = layer(x, key_padding_mask=key_padding_mask) # Shape: [batch_size, seq_len, d_model]

        cls_out = x[:, 0, :] # Extract CLS token output: Shape: [batch_size, d_model]
        return x, cls_out # Return all token outputs and CLS token output
    
