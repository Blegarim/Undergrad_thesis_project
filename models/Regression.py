import torch
import torch.nn as nn
import torch.nn.functional as F

class MotionEncoder(nn.Module):
    def __init__(self, 
                 input_dim=3, 
                 tcn_channels=(64, 128), 
                 d_model=128, 
                 num_layers=2, 
                 kernel_size=3,
                 num_heads=8, 
                 dropout=0.3):
        super().__init__()
        # --- Temporal Convolutional Network (TCN) ---
        layers = []
        in_channels = input_dim
        for out_channels in tcn_channels:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding='same'))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        self.tcn = nn.Sequential(*layers)

        # --- Gated Recurrent Unit (GRU) ---
        self.gru = nn.GRU(input_size=tcn_channels[-1], hidden_size=tcn_channels[-1], 
                          num_layers=num_layers, batch_first=True, 
                          dropout=dropout if num_layers > 1 else 0)
        
        self.proj = nn.Linear(tcn_channels[-1], d_model if tcn_channels[-1] != d_model else nn.Identity)
        
        # --- Attention ---
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.num_heads = num_heads
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: [B, T, C]
        B, T, C = x.shape
        assert C == self.tcn[0].in_channels, f"Input feature dimension {C} does not match expected {self.tcn[0].in_channels}"
        x = x.transpose(1, 2) # Shape: [B, C, T]
        x = self.tcn(x)       # Shape: [B, D_tcn, T]
        x = x.transpose(1, 2) # Shape: [B, T, D_tcn]
        x, _ = self.gru(x) # Shape: [B, T, D_tcn]
        x = self.proj(x) # Shape: [B, T, d_model]
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        B, T, D = q.shape
        H = self.num_heads
        assert D % H == 0, f"Input dimension {D} not divisible by num_heads {H}"
        Dh = D // H
        q = q.view(B, T, H, Dh).transpose(1, 2) # Shape: [B, H, T, Dh]
        k = k.view(B, T, H, Dh).transpose(1, 2) # Shape: [B, H, T, Dh]
        v = v.view(B, T, H, Dh).transpose(1, 2) # Shape: [B, H, T, Dh]
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p)
        attn = attn_out.transpose(1, 2).contiguous().view(B, T, D)
        x = self.dropout(attn)
        return x
    
if __name__ == "__main__":
    # Test the M module
    batch_size = 8
    seq_len = 50
    input_dim = 4
    x = torch.randn(batch_size, seq_len, input_dim) # Example input

    model = MotionEncoder(input_dim=input_dim, tcn_channels=(128, 256), d_model=128, num_layers=2, kernel_size=3, num_heads=8, dropout=0.1)
    out = model(x)
    print("Output shape:", out.shape) # Expected: [batch_size, seq_len, d_model]
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")