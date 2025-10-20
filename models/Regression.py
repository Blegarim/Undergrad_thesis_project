import torch
import torch.nn as nn
import torch.nn.functional as F

class TCNGRU(nn.Module):
    def __init__(self, input_dim=3, 
                 tcn_channels=(64, 128), d_model=128, 
                 num_layers=2, kernel_size=3, 
                 dropout=0.1):
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
        d_model = tcn_channels[-1] # Ensure d_model matches the last TCN output channels
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, 
                          num_layers=num_layers, batch_first=True, 
                          dropout=dropout if num_layers > 1 else 0)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: [B, T, C]
        B, T, C = x.shape
        assert C == self.tcn[0].in_channels, f"Input feature dimension {C} does not match expected {self.tcn[0].in_channels}"
        x = x.transpose(1, 2) # Shape: [B, C, T]
        x = self.tcn(x)       # Shape: [B, D, T]
        x = x.transpose(1, 2) # Shape: [B, T, D]
        x, _ = self.gru(x) # Shape: [B, T, D]
        x = self.dropout(x)
        return x
    
if __name__ == "__main__":
    # Test the TCNGRU module
    batch_size = 8
    seq_len = 50
    input_dim = 4
    x = torch.randn(batch_size, seq_len, input_dim) # Example input

    model = TCNGRU(input_dim=input_dim, tcn_channels=(64, 128), d_model=256, num_layers=2, kernel_size=3, dropout=0.1)
    out = model(x)
    print("Output shape:", out.shape) # Expected: [batch_size, seq_len, d_model]